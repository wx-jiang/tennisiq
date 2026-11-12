"""Score the saved Elo baseline and XGBoost model on the held-out 2025 season.

Reproduces every number in artifacts/model_metrics.json and the calibration plot.
Run after `python -m src.train`.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from src.data import load_matches
from src.features import build_player_features
from src.train import ARTIFACTS, META_PATH, MODEL_PATH, TEST_YEAR, elo_win_prob, xgb_win_prob

METRICS_PATH = ARTIFACTS / "model_metrics.json"
CALIBRATION_PATH = ARTIFACTS / "calibration.png"
N_BOOTSTRAP = 2000


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y, p > 0.5),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "roc_auc": roc_auc_score(y, p),
    }


def paired_bootstrap(p_winner_a: np.ndarray, p_winner_b: np.ndarray, seed: int = 0) -> dict:
    """95% CI for (log loss A - log loss B), resampling matches with replacement."""
    rng = np.random.default_rng(seed)
    loss_a, loss_b = -np.log(p_winner_a), -np.log(p_winner_b)
    idx = rng.integers(0, len(loss_a), size=(N_BOOTSTRAP, len(loss_a)))
    diffs = loss_a[idx].mean(axis=1) - loss_b[idx].mean(axis=1)
    return {"mean_diff": float((loss_a - loss_b).mean()),
            "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]}


def plot_calibration(y: np.ndarray, probs: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="Perfect calibration")
    for (name, p), color in zip(probs.items(), ["#1f77b4", "#d62728"], strict=True):
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", color=color, label=name)
    ax.set_xlabel("Predicted P(player A wins)")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration on {TEST_YEAR} ATP matches")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(CALIBRATION_PATH, dpi=150)


def main() -> None:
    meta = json.loads(META_PATH.read_text())
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    df, _ = build_player_features(load_matches())
    test = df[df["tourney_date"].dt.year == TEST_YEAR]

    p_win = {"Elo": elo_win_prob(test, meta["elo_weight"]), "XGBoost": xgb_win_prob(model, test)}

    # Both models are orientation-invariant, so accuracy / log loss / Brier do not depend
    # on which player is "A". AUC needs both classes, so we orient by a seeded coin flip.
    a_is_winner = np.random.default_rng(42).random(len(test)) < 0.5
    y = a_is_winner.astype(int)
    p_a = {name: np.where(a_is_winner, p, 1 - p) for name, p in p_win.items()}

    higher_ranked_won = (test["winner_rank"].fillna(9999) < test["loser_rank"].fillna(9999)).mean()
    results = {
        "test_period": {"start": test["tourney_date"].min().date().isoformat(),
                        "end": test["tourney_date"].max().date().isoformat(),
                        "n_matches": int(len(test))},
        "model_version": meta["model_version"],
        "models": {name: metrics(y, p) for name, p in p_a.items()},
        "reference_higher_ranked_wins_accuracy": float(higher_ranked_won),
        "xgb_minus_elo_log_loss": paired_bootstrap(p_win["XGBoost"], p_win["Elo"]),
    }
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    plot_calibration(y, p_a)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
