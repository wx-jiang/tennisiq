import pytest

from src.predict import PlayerNotFound, predict, resolve_player
from src.tools import execute_tool, get_player_stats, validate_args


def test_prediction_is_orientation_invariant():
    ab = predict("Carlos Alcaraz", "Jannik Sinner", "Clay")
    ba = predict("Jannik Sinner", "Carlos Alcaraz", "Clay")
    assert ab["xgboost_win_prob"]["Carlos Alcaraz"] == pytest.approx(ba["xgboost_win_prob"]["Carlos Alcaraz"], abs=1e-3)
    assert sum(ab["xgboost_win_prob"].values()) == pytest.approx(1.0, abs=1e-3)
    assert sum(ab["elo_win_prob"].values()) == pytest.approx(1.0, abs=1e-3)


def test_prediction_reports_cutoff_and_version():
    result = predict("Sinner", "Zverev", "hard")
    assert result["data_cutoff"] and result["model_version"]
    assert result["surface"] == "Hard"


def test_resolve_player_variants():
    assert resolve_player("alcaraz")["name"] == "Carlos Alcaraz"
    assert resolve_player("Felix Auger-Aliassime")["name"] == "Felix Auger Aliassime"


def test_unknown_player_raises():
    with pytest.raises(PlayerNotFound):
        resolve_player("Roger Rabbit")


def test_stale_player_gets_warning():
    result = predict("Roger Federer", "Jannik Sinner", "Grass")
    assert any("Federer" in w for w in result["warnings"])


def test_player_stats_add_up():
    by_surface = [get_player_stats("Novak Djokovic", 2024, s) for s in ("Hard", "Clay", "Grass")]
    total = get_player_stats("Novak Djokovic", 2024)
    assert total["wins"] == sum(s["wins"] for s in by_surface)
    assert total["matches"] == total["wins"] + total["losses"]


@pytest.mark.parametrize("name,args,message", [
    ("predict_match", {"player1": "Alcaraz", "player2": "Sinner", "surface": "Carpet"}, "Unsupported surface"),
    ("predict_match", {"player1": "Alcaraz", "player2": "Alcaraz", "surface": "Clay"}, "different players"),
    ("predict_match", {"player1": "Alcaraz"}, "Missing required"),
    ("get_player_stats", {"player": "Sinner", "year": 2031}, "outside the available data"),
    ("get_player_stats", {"player": "Sinner", "code": "import os"}, "Unknown argument"),
    ("run_python", {"code": "print(1)"}, "Unknown tool"),
])
def test_execute_tool_returns_errors(name, args, message):
    assert message in execute_tool(name, args)["error"]


def test_validate_args_coerces_year_and_drops_nulls():
    assert validate_args("get_player_stats", {"player": "Sinner", "year": "2024", "surface": None}) == {
        "player": "Sinner", "year": 2024}
    with pytest.raises(ValueError):
        validate_args("get_player_stats", {"player": 42})
