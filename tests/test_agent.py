import json

from evaluation.evaluate_agent import matches_value, ungrounded_numbers
from src import agent
from src.llm import LLMReply


def fake_llm(replies):
    calls = []

    def chat(messages, tools):
        calls.append(messages)
        return replies[min(len(calls) - 1, len(replies) - 1)]

    return chat, calls


def test_agent_executes_tool_then_answers(monkeypatch):
    chat, calls = fake_llm([
        LLMReply("", [{"id": "c1", "name": "get_player_stats",
                       "arguments": {"player": "Djokovic", "year": 2024, "surface": "Hard"}}]),
        LLMReply("Djokovic won 75.0% of hard-court matches in 2024."),
    ])
    monkeypatch.setattr(agent.llm, "chat", chat)
    result = agent.run_agent("Djokovic hard 2024?")
    assert result.iterations == 2
    assert result.tool_calls[0]["result"]["win_pct"] == 75.0
    tool_msg = calls[1][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["wins"] == 18


def test_agent_stops_after_max_iterations(monkeypatch):
    looping = LLMReply("", [{"id": "c", "name": "compare_surfaces", "arguments": {"player1": "Sinner", "player2": "Ruud"}}])
    chat, calls = fake_llm([looping])
    monkeypatch.setattr(agent.llm, "chat", chat)
    result = agent.run_agent("loop forever")
    assert len(calls) == agent.MAX_ITERATIONS
    assert "tool-call limit" in result.answer


def test_agent_passes_tool_errors_back(monkeypatch):
    chat, calls = fake_llm([
        LLMReply("", [{"id": "c1", "name": "delete_files", "arguments": {}}]),
        LLMReply("I can't do that."),
    ])
    monkeypatch.setattr(agent.llm, "chat", chat)
    result = agent.run_agent("delete everything")
    assert "Unknown tool" in result.tool_calls[0]["result"]["error"]


def test_history_carries_over(monkeypatch):
    chat, calls = fake_llm([LLMReply("first"), LLMReply("second")])
    monkeypatch.setattr(agent.llm, "chat", chat)
    first = agent.run_agent("q1")
    agent.run_agent("q2", first.history)
    assert [m["content"] for m in calls[1] if m["role"] == "user"] == ["q1", "q2"]


def test_grounding_check():
    assert matches_value(59.6, 0.596) and matches_value(60, 0.596) and not matches_value(70, 0.596)
    history = [{"role": "tool", "content": json.dumps({"p": 0.596, "wins": 18})}]
    assert ungrounded_numbers("Alcaraz 59.6%, 18 wins, cutoff 2025-12-17", history, "") == []
    assert ungrounded_numbers("He has a 72% chance", history, "") == [72.0]
