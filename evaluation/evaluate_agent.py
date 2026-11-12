"""Benchmark the tool-calling agent on evaluation/test_queries.json.

Criteria (all programmatic; the LLM's own wording is never trusted as evidence):

1. Tool selection: the set of tool names the agent called equals the set in one of the
   case's acceptable plans (an empty plan means "answer without tools").
2. Argument correctness: fraction of the matched plan's expected calls for which the agent
   made a call with the same tool and equivalent arguments. Player names are compared after
   resolving to the canonical player (order-insensitive for two-player tools), surfaces
   case-insensitively, years as integers. "*" accepts any arguments.
3. Numerical correctness, two parts:
   - fact recall: each expected fact (ground truth computed by calling the tool directly)
     appears in the answer, as a probability (0.596), a percentage (59.6 or 60) or a count;
   - grounding: every number in the answer appears in a tool result of the conversation or
     in the user's question (years, ISO dates and the constants 0-3, 5, 20 are ignored).
4. End-to-end success: correct tool selection, all arguments correct, all facts recalled and
   no ungrounded numbers. For invalid-input cases there are no facts, and the answer must
   additionally contain no percentage (i.e. no invented prediction or win rate).
5. Latency: wall-clock seconds per query (setup turns for follow-ups excluded).
"""

import json
import re
import statistics
from pathlib import Path

from src import llm
from src.agent import run_agent
from src.predict import PlayerNotFound, resolve_player
from src.tools import execute_tool
from src.train import ARTIFACTS

HERE = Path(__file__).resolve().parent
QUERIES_PATH = HERE / "test_queries.json"
RESULTS_PATH = HERE / "results" / "agent_eval_results.json"
SUMMARY_PATH = ARTIFACTS / "agent_eval_summary.json"

IGNORED_NUMBERS = {0, 1, 2, 3, 5, 20}
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def canonical_player(name: str) -> str:
    try:
        return resolve_player(name)["name"]
    except PlayerNotFound:
        return str(name).strip().lower()


def canonical_args(name: str, args: dict) -> tuple:
    if not isinstance(args, dict):
        return ("invalid",)
    clean = {k: v for k, v in args.items() if v not in (None, "")}
    if str(clean.get("surface", "")).lower() in ("all", "any", "overall"):
        clean.pop("surface")
    players = frozenset(canonical_player(clean.pop(k)) for k in ("player", "player1", "player2") if k in clean)
    surface = str(clean.pop("surface", "")).capitalize() or None
    year = clean.pop("year", None)
    year = int(year) if isinstance(year, (int, str)) and str(year).isdigit() else year
    return (name, players, surface, year, tuple(sorted(clean.items())))


def score_plan(plan: list[dict], calls: list[dict]) -> tuple[bool, float]:
    selection_ok = {c["name"] for c in calls} == {c["name"] for c in plan}
    if not plan:
        return selection_ok, 1.0
    actual = {canonical_args(c["name"], c["arguments"]) for c in calls}
    hits = sum(
        any(c["name"] == e["name"] for c in calls) if e["arguments"] == "*"
        else canonical_args(e["name"], e["arguments"]) in actual
        for e in plan
    )
    return selection_ok, hits / len(plan)


def numbers_in(text: str) -> list[float]:
    return [float(x) for x in NUMBER_RE.findall(DATE_RE.sub(" ", text))]


def is_ignorable(x: float) -> bool:
    return (x.is_integer() and 1900 <= x <= 2100) or x in IGNORED_NUMBERS


def json_numbers(obj) -> set[float]:
    if isinstance(obj, bool):
        return set()
    if isinstance(obj, (int, float)):
        return {float(obj)}
    if isinstance(obj, dict):
        return set().union(*(json_numbers(v) for v in obj.values()), *(numbers_in(k) for k in obj)) if obj else set()
    if isinstance(obj, list):
        return set().union(*(json_numbers(v) for v in obj)) if obj else set()
    if isinstance(obj, str):
        return set(numbers_in(obj))
    return set()


def matches_value(x: float, value: float) -> bool:
    """Answer number x expresses `value`: probabilities may appear as 0.596, 59.6% or 60%."""
    targets = [value, value * 100] if 0 < value < 1 else [value]
    return any(abs(x - t) < 0.051 or (x.is_integer() and x == round(t)) for t in targets)


def lookup(result: dict, path: str):
    for key in path.split("."):
        result = result[key]
    return result


def fact_recall(case: dict, answer: str) -> tuple[int, int]:
    nums = numbers_in(answer)
    found = 0
    for fact in case["facts"]:
        truth = execute_tool(fact["tool"], fact["args"])
        values = [lookup(truth, p) for p in fact["any_of"]]
        found += any(matches_value(x, v) for x in nums for v in values if v is not None)
    return found, len(case["facts"])


def ungrounded_numbers(answer: str, history: list[dict], question: str) -> list[float]:
    grounded = set(numbers_in(question))
    for m in history:
        if m["role"] == "tool":
            grounded |= json_numbers(json.loads(m["content"]))
        elif m["role"] == "user":
            grounded |= set(numbers_in(m["content"]))
    return [x for x in numbers_in(answer)
            if not is_ignorable(x) and not any(matches_value(x, g) for g in grounded)]


def evaluate_case(case: dict) -> dict:
    history: list[dict] = []
    for setup_question in case.get("setup", []):
        history = run_agent(setup_question, history).history
    result = run_agent(case["query"], history)

    best = max((score_plan(plan, result.tool_calls) for plan in case["plans"]), key=lambda s: (s[0], s[1]))
    selection_ok, arg_score = best
    found, total = fact_recall(case, result.answer)
    ungrounded = ungrounded_numbers(result.answer, result.history, case["query"])
    success = selection_ok and arg_score == 1.0 and found == total and not ungrounded
    if case["category"] == "invalid":
        success = success and "%" not in result.answer
    return {
        "id": case["id"], "category": case["category"], "query": case["query"],
        "answer": result.answer,
        "tool_calls": [{"name": c["name"], "arguments": c["arguments"], "error": c["result"].get("error")}
                       for c in result.tool_calls],
        "tool_selection_correct": selection_ok,
        "argument_score": arg_score,
        "facts_found": found, "facts_total": total,
        "ungrounded_numbers": ungrounded,
        "success": success,
        "iterations": result.iterations,
        "latency_s": round(result.latency_s, 2),
    }


def summarize(rows: list[dict]) -> dict:
    latencies = sorted(r["latency_s"] for r in rows)
    facts_total = sum(r["facts_total"] for r in rows)
    by_cat = {}
    for cat in dict.fromkeys(r["category"] for r in rows):
        sub = [r for r in rows if r["category"] == cat]
        by_cat[cat] = f"{sum(r['success'] for r in sub)}/{len(sub)}"
    return {
        "provider": llm.provider_name(), "model": llm.model_name(), "n_queries": len(rows),
        "tool_selection_accuracy": round(statistics.mean(r["tool_selection_correct"] for r in rows), 3),
        "argument_accuracy": round(statistics.mean(r["argument_score"] for r in rows), 3),
        "fact_recall": round(sum(r["facts_found"] for r in rows) / facts_total, 3),
        "answers_with_ungrounded_numbers": sum(bool(r["ungrounded_numbers"]) for r in rows),
        "end_to_end_success_rate": round(statistics.mean(r["success"] for r in rows), 3),
        "success_by_category": by_cat,
        "latency_s": {"mean": round(statistics.mean(latencies), 2), "median": round(statistics.median(latencies), 2),
                      "p90": latencies[int(0.9 * (len(latencies) - 1))]},
    }


def main() -> None:
    if not llm.is_configured():
        raise SystemExit("No LLM configured: start Ollama or set LLM_PROVIDER=openai and OPENAI_API_KEY.")
    cases = json.loads(QUERIES_PATH.read_text())
    rows = []
    for case in cases:
        rows.append(evaluate_case(case))
        r = rows[-1]
        print(f"{r['id']:<11} success={r['success']!s:<5} tools={[c['name'] for c in r['tool_calls']]} "
              f"latency={r['latency_s']}s")
    summary = summarize(rows)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(rows, indent=2))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
