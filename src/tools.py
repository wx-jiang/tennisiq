"""The three tools exposed to the LLM, their JSON schemas, and a validating dispatcher."""

from typing import Any

from src.data import SURFACES
from src.predict import PlayerNotFound, load_model, load_stats, predict, resolve_player, validate_surface

STATS_YEARS = range(2010, 2026)
STATS_SCOPE = ("ATP tour-level main-draw singles (incl. Grand Slams, Masters, ATP 250/500, Finals, "
               "Davis Cup, Olympics), walkovers excluded. Counts may differ slightly from official ATP records.")


def predict_match(player1: str, player2: str, surface: str) -> dict:
    result = predict(player1, player2, surface)
    result["note"] = ("Model-calculated probability (not a historical fact), based on features "
                      f"frozen at the data cutoff {result['data_cutoff']}. Assumes best-of-3.")
    return result


def get_player_stats(player: str, year: int | None = None, surface: str | None = None) -> dict:
    p = resolve_player(player)
    if year is not None and year not in STATS_YEARS:
        raise ValueError(f"Year {year} is outside the available data (2010-2025).")
    if surface is not None and surface.strip().lower() in ("", "all", "any", "overall"):
        surface = None
    if surface is not None:
        surface = validate_surface(surface)

    stats = load_stats()
    rows = stats[stats["player_id"] == p["player_id"]]
    if year is not None:
        rows = rows[rows["year"] == year]
    if surface is not None:
        rows = rows[rows["surface"] == surface]
    wins, losses = int(rows["wins"].sum()), int(rows["losses"].sum())
    played = wins + losses
    return {
        "player": p["name"],
        "year": year if year is not None else "2010-2025",
        "surface": surface or "All",
        "matches": played,
        "wins": wins,
        "losses": losses,
        "win_pct": round(100 * wins / played, 1) if played else None,
        "scope": STATS_SCOPE,
        "note": "Historical fact computed from match records." if played else "No matches found for this filter.",
    }


def compare_surfaces(player1: str, player2: str) -> dict:
    by_surface = {s: predict(player1, player2, s) for s in SURFACES}
    first = by_surface[SURFACES[0]]
    return {
        "player1": first["player1"],
        "player2": first["player2"],
        "xgboost_win_prob_by_surface": {s: r["xgboost_win_prob"] for s, r in by_surface.items()},
        "elo_win_prob_by_surface": {s: r["elo_win_prob"] for s, r in by_surface.items()},
        "surface_elo": {s: {k: v for k, v in r["elo"].items() if isinstance(v, dict)} for s, r in by_surface.items()},
        "model_version": first["model_version"],
        "data_cutoff": first["data_cutoff"],
        "warnings": first["warnings"],
        "note": "Model-calculated probabilities (not historical facts). Assumes best-of-3.",
    }


TOOL_FUNCTIONS = {"predict_match": predict_match, "get_player_stats": get_player_stats,
                  "compare_surfaces": compare_surfaces}

_PLAYER = {"type": "string", "description": "Player name, e.g. 'Carlos Alcaraz' or 'Sinner'."}
_SURFACE = {"type": "string", "enum": list(SURFACES), "description": "Court surface."}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "predict_match",
        "description": "Predict the win probability for a hypothetical match between two ATP players on a surface.",
        "parameters": {"type": "object", "properties": {"player1": _PLAYER, "player2": _PLAYER, "surface": _SURFACE},
                       "required": ["player1", "player2", "surface"]}}},
    {"type": "function", "function": {
        "name": "get_player_stats",
        "description": ("Historical win/loss record of an ATP player (2010-2025), optionally filtered by "
                        "year and surface. Omit year for all years; omit surface for all surfaces."),
        "parameters": {"type": "object", "properties": {
            "player": _PLAYER,
            "year": {"type": "integer", "description": "Season, 2010-2025."},
            "surface": _SURFACE},
            "required": ["player"]}}},
    {"type": "function", "function": {
        "name": "compare_surfaces",
        "description": "Predicted win probabilities for two players on hard, clay and grass courts.",
        "parameters": {"type": "object", "properties": {"player1": _PLAYER, "player2": _PLAYER},
                       "required": ["player1", "player2"]}}},
]


def validate_args(name: str, args: Any) -> dict:
    """Check tool arguments against the schema; drop nulls, coerce numeric strings for year."""
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    schema = next(t["function"]["parameters"] for t in TOOL_SCHEMAS if t["function"]["name"] == name)
    props = schema["properties"]
    unknown = set(args) - set(props)
    if unknown:
        raise ValueError(f"Unknown argument(s) for {name}: {sorted(unknown)}")
    clean = {k: v for k, v in args.items() if v not in (None, "")}
    missing = [k for k in schema["required"] if k not in clean]
    if missing:
        raise ValueError(f"Missing required argument(s) for {name}: {missing}")
    for key, value in clean.items():
        expected = props[key]["type"]
        if expected == "integer":
            if isinstance(value, str) and value.strip().isdigit():
                value = int(value)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"Argument '{key}' must be an integer.")
        elif not isinstance(value, str):
            raise ValueError(f"Argument '{key}' must be a string.")
        clean[key] = value
    return clean


def execute_tool(name: str, args: Any) -> dict:
    """Run a whitelisted tool. User-facing problems (bad names, surfaces, years) come
    back as {'error': ...} so the LLM can explain or retry; anything else propagates."""
    if name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool '{name}'. Available: {sorted(TOOL_FUNCTIONS)}"}
    try:
        return TOOL_FUNCTIONS[name](**validate_args(name, args))
    except (PlayerNotFound, ValueError) as exc:
        return {"error": str(exc)}


def data_cutoff() -> str:
    return load_model()[1]["data_cutoff"]
