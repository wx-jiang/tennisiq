"""Bounded tool-calling agent loop on top of src.llm and src.tools."""

import json
import time
from dataclasses import dataclass, field

from src import llm
from src.tools import TOOL_SCHEMAS, data_cutoff, execute_tool

MAX_ITERATIONS = 5

SYSTEM_PROMPT = """You are TennisIQ, an ATP tennis analyst. You answer using tools only.

Rules:
- Every number you state (probabilities, Elo ratings, win/loss records, percentages) must come
  verbatim from a tool result in this conversation. Never estimate, recall from memory, or
  compute numbers yourself. If no tool provides a number, say you don't have it.
- Use predict_match for a single head-to-head on one surface, compare_surfaces when the user
  asks about several or all surfaces, and get_player_stats for historical records.
- If the user does not name a surface for a prediction, use compare_surfaces.
- For follow-up questions ("what about on grass?"), reuse the players from earlier turns.
- Clearly label predictions as model estimates and records as historical facts.
- Predictions use data up to {cutoff}. Mention this cutoff when giving a prediction, and relay
  any warnings from the tool (e.g. outdated player data).
- If a tool returns an error, explain it plainly; you may retry once with corrected arguments
  (for example a full player name). Do not invent data for unknown players.
- Only ATP men's singles 2010-2025 data is available. Politely decline unrelated requests.
- Be concise: 2-5 sentences, plain text, no tables."""


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    iterations: int = 0
    latency_s: float = 0.0
    history: list[dict] = field(default_factory=list)


def system_message() -> dict:
    return {"role": "system", "content": SYSTEM_PROMPT.format(cutoff=data_cutoff())}


def run_agent(question: str, history: list[dict] | None = None) -> AgentResult:
    """Answer `question`. `history` is the prior conversation (without the system message);
    the returned `history` includes this turn and can be passed back for follow-ups."""
    start = time.perf_counter()
    turn = [*(history or []), {"role": "user", "content": question}]
    executed: list[dict] = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        reply = llm.chat([system_message(), *turn], TOOL_SCHEMAS)
        turn.append({"role": "assistant", "content": reply.content, "tool_calls": reply.tool_calls})
        if not reply.tool_calls:
            return AgentResult(reply.content.strip(), executed, iteration, time.perf_counter() - start, turn)

        for call in reply.tool_calls:
            result = execute_tool(call["name"], call["arguments"])
            executed.append({"name": call["name"], "arguments": call["arguments"], "result": result})
            turn.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                         "content": json.dumps(result)})

    answer = ("I couldn't complete this request within the tool-call limit. "
              "Please try a more specific question.")
    turn.append({"role": "assistant", "content": answer, "tool_calls": []})
    return AgentResult(answer, executed, MAX_ITERATIONS, time.perf_counter() - start, turn)
