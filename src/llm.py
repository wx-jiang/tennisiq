"""Minimal LLM provider layer: local Ollama or any OpenAI-compatible chat API.

Messages use one internal format, converted per provider:
  {"role": "system" | "user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "arguments": dict, "raw"?}]}
  ("raw" is the OpenAI-format call exactly as the provider returned it, replayed unchanged.)
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}

Configuration (environment variables; the Streamlit app also copies st.secrets into env):
  LLM_PROVIDER     "ollama" (default) or "openai"
  OLLAMA_MODEL     default "qwen3:14b";   OLLAMA_HOST optional
  OPENAI_API_KEY   required for "openai"; OPENAI_MODEL default "gpt-4o-mini"
  OPENAI_BASE_URL  optional, for OpenAI-compatible providers (Groq, Together, OpenRouter, ...)
"""

import json
import os
from dataclasses import dataclass, field


@dataclass
class LLMReply:
    content: str
    tool_calls: list[dict] = field(default_factory=list)


def provider_name() -> str:
    return os.getenv("LLM_PROVIDER", "ollama").strip().lower()


def model_name() -> str:
    if provider_name() == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return os.getenv("OLLAMA_MODEL", "qwen3:14b")


def is_configured() -> bool:
    if provider_name() == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    return provider_name() == "ollama" and _ollama_reachable()


def chat(messages: list[dict], tools: list[dict]) -> LLMReply:
    provider = provider_name()
    if provider == "ollama":
        return _chat_ollama(messages, tools)
    if provider == "openai":
        return _chat_openai(messages, tools)
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use 'ollama' or 'openai'.")


def _ollama_reachable() -> bool:
    import ollama

    try:
        ollama.Client(host=os.getenv("OLLAMA_HOST")).list()
    except ConnectionError:
        return False
    return True


def _chat_ollama(messages: list[dict], tools: list[dict]) -> LLMReply:
    import ollama

    converted = []
    for m in messages:
        if m["role"] == "assistant":
            calls = [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in m.get("tool_calls", [])]
            converted.append({"role": "assistant", "content": m["content"], "tool_calls": calls})
        elif m["role"] == "tool":
            converted.append({"role": "tool", "content": m["content"], "tool_name": m["name"]})
        else:
            converted.append(m)

    client = ollama.Client(host=os.getenv("OLLAMA_HOST"))
    resp = client.chat(model=model_name(), messages=converted, tools=tools, think=False,
                       options={"temperature": 0, "seed": 0})
    calls = [
        {"id": f"call_{i}", "name": c.function.name, "arguments": dict(c.function.arguments or {})}
        for i, c in enumerate(resp.message.tool_calls or [])
    ]
    return LLMReply(content=resp.message.content or "", tool_calls=calls)


def _chat_openai(messages: list[dict], tools: list[dict]) -> LLMReply:
    from openai import OpenAI

    converted = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = [c.get("raw") or {"id": c["id"], "type": "function",
                                      "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                     for c in m["tool_calls"]]
            converted.append({"role": "assistant", "content": m["content"] or None, "tool_calls": calls})
        elif m["role"] == "assistant":
            converted.append({"role": "assistant", "content": m["content"]})
        elif m["role"] == "tool":
            converted.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            converted.append(m)

    # Extra retries ride out short rate-limit (429) bursts on free API tiers.
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.getenv("OPENAI_BASE_URL") or None,
                    max_retries=5)
    resp = client.chat.completions.create(model=model_name(), messages=converted, tools=tools, temperature=0)
    msg = resp.choices[0].message
    calls = []
    for c in msg.tool_calls or []:
        try:
            args = json.loads(c.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {"_invalid_json": c.function.arguments}  # rejected later by argument validation
        # Keep the provider's tool call verbatim for replay: Gemini 3 attaches a thought
        # signature (extra_content) that must be sent back or the next request fails.
        calls.append({"id": c.id, "name": c.function.name, "arguments": args,
                      "raw": c.model_dump(exclude_none=True)})
    return LLMReply(content=msg.content or "", tool_calls=calls)
