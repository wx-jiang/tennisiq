import json
import os

import pandas as pd
import streamlit as st

from src import llm
from src.agent import run_agent
from src.predict import MIN_MATCHES, player_index
from src.train import ARTIFACTS
from src.tools import compare_surfaces, data_cutoff, predict_match

SECRET_KEYS = ("LLM_PROVIDER", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "OLLAMA_MODEL", "OLLAMA_HOST")

st.set_page_config(page_title="TennisIQ", page_icon="🎾", layout="wide")


def load_secrets_into_env() -> None:
    try:
        secrets = {k: str(st.secrets[k]) for k in SECRET_KEYS if k in st.secrets}
    except FileNotFoundError:  # no secrets.toml: rely on environment variables
        return
    for key, value in secrets.items():
        os.environ.setdefault(key, value)


@st.cache_data
def player_options() -> list[str]:
    players = player_index()
    players = players[players["matches"] >= MIN_MATCHES]
    recent = players[players["last_match"] >= "2025-01-01"].sort_values("matches", ascending=False)
    return recent["name"].tolist() + sorted(set(players["name"]) - set(recent["name"]))


@st.cache_data(ttl=60)
def llm_available() -> bool:
    return llm.is_configured()


def read_json(name: str) -> dict | None:
    path = ARTIFACTS / name
    return json.loads(path.read_text()) if path.exists() else None


def prob_bar(probs: dict[str, float]) -> None:
    (a, pa), (b, pb) = probs.items()
    c1, c2 = st.columns(2)
    c1.metric(a, f"{pa:.1%}")
    c2.metric(b, f"{pb:.1%}")
    st.progress(pa)


def prediction_tab() -> None:
    names = player_options()
    c1, c2, c3 = st.columns([2, 2, 1])
    p1 = c1.selectbox("Player A", names, index=names.index("Carlos Alcaraz") if "Carlos Alcaraz" in names else 0)
    p2 = c2.selectbox("Player B", names, index=names.index("Jannik Sinner") if "Jannik Sinner" in names else 1)
    surface = c3.radio("Surface", ["Hard", "Clay", "Grass"], horizontal=False)
    if p1 == p2:
        st.warning("Choose two different players.")
        return

    result = predict_match(p1, p2, surface)
    for w in result["warnings"]:
        st.warning(w)

    left, right = st.columns(2)
    with left:
        st.subheader("XGBoost win probability")
        prob_bar(result["xgboost_win_prob"])
    with right:
        st.subheader("Elo baseline win probability")
        prob_bar(result["elo_win_prob"])

    st.subheader("Elo ratings")
    elo = pd.DataFrame({name: result["elo"][name] for name in (p1, p2)}).T
    elo.columns = ["Overall Elo", f"{surface} Elo"]
    st.dataframe(elo, width="stretch")

    st.subheader("Model inputs (as of data cutoff)")
    inputs = pd.DataFrame(result["inputs"]).rename(index=lambda s: s.replace("_", " ").capitalize())
    st.dataframe(inputs.astype(str), width="stretch")
    st.caption("Win rates are shrunk toward 50% (Beta(2,2) prior) so that small samples are not extreme. "
               "Assumes best-of-3. Head-to-head record is not a model input.")

    st.subheader("Across surfaces")
    surfaces = compare_surfaces(p1, p2)["xgboost_win_prob_by_surface"]
    table = pd.DataFrame({"Surface": list(surfaces), f"P({p1} wins)": [v[p1] for v in surfaces.values()]})
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        f"P({p1} wins)": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0)})


def chat_tab() -> None:
    if not llm_available():
        st.info(
            "The AI analyst needs an LLM. Locally, run Ollama (`ollama serve`, `ollama pull qwen3:14b`). "
            "On the hosted app, set `LLM_PROVIDER = \"openai\"` and `OPENAI_API_KEY` in Streamlit secrets "
            "(any OpenAI-compatible API works via `OPENAI_BASE_URL`). Match prediction and model "
            "performance work without an LLM."
        )
        return

    st.caption(f"Provider: {llm.provider_name()} · model: {llm.model_name()} · "
               "every number comes from a Python tool call shown under the answer.")
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("turns", [])
    if st.button("Clear conversation"):
        st.session_state.history, st.session_state.turns = [], []

    examples = ["Who is more likely to win between Alcaraz and Sinner on clay?",
                "What was Djokovic's win rate on hard courts in 2024?",
                "How would the prediction change if they played on grass instead?"]
    st.markdown("**Try:** " + " · ".join(f"_{q}_" for q in examples))

    for turn in st.session_state.turns:
        render_turn(turn)

    question = st.chat_input("Ask about ATP players, matchups or surfaces")
    if not question:
        return
    with st.spinner("Thinking and calling tools..."):
        result = run_agent(question, st.session_state.history)
    st.session_state.history = result.history
    turn = {"question": question, "answer": result.answer, "tool_calls": result.tool_calls,
            "latency": result.latency_s}
    st.session_state.turns.append(turn)
    render_turn(turn)


def render_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        label = f"{len(turn['tool_calls'])} tool call(s) · {turn['latency']:.1f}s"
        with st.expander(label):
            for call in turn["tool_calls"]:
                st.markdown(f"`{call['name']}({json.dumps(call['arguments'])})`")
                st.json(call["result"], expanded=False)


def performance_tab() -> None:
    metrics = read_json("model_metrics.json")
    meta = read_json("model_meta.json")
    tp = metrics["test_period"]
    st.markdown(f"Held-out test set: **{tp['n_matches']:,} ATP tour-level matches, {tp['start']} to {tp['end']}**. "
                f"Models were trained on {meta['trained_on']}; hyperparameters were chosen on "
                f"{', '.join(map(str, meta['validation_years']))} validation folds, never on the test year.")
    table = pd.DataFrame(metrics["models"]).T
    table.columns = ["Accuracy", "Log loss", "Brier score", "ROC-AUC"]
    st.dataframe(table.style.format("{:.3f}"), width="stretch")
    ci = metrics["xgb_minus_elo_log_loss"]
    st.caption(f"XGBoost minus Elo log loss: {ci['mean_diff']:.4f} "
               f"(paired bootstrap 95% CI {ci['ci95'][0]:.4f} to {ci['ci95'][1]:.4f}). "
               f"Reference: the higher-ranked player won {metrics['reference_higher_ranked_wins_accuracy']:.1%} of test matches.")

    c1, c2 = st.columns([1, 1])
    c1.image(str(ARTIFACTS / "calibration.png"))
    agent = read_json("agent_eval_summary.json")
    with c2:
        st.subheader("Agent benchmark")
        if agent:
            st.json(agent)
        else:
            st.write("Run `python -m evaluation.evaluate_agent` to populate.")


def main() -> None:
    load_secrets_into_env()
    st.title("🎾 TennisIQ")
    st.markdown("ATP match prediction (XGBoost vs. Elo) with an LLM analyst that answers through Python tools.")
    st.info(f"**Data cutoff: {data_cutoff()}.** Predictions use player features frozen at this date and are "
            "not live forecasts; recent form, injuries and ranking changes after the cutoff are not reflected.")
    tabs = st.tabs(["Match Prediction", "AI Tennis Analyst", "Model Performance"])
    with tabs[0]:
        prediction_tab()
    with tabs[1]:
        chat_tab()
    with tabs[2]:
        performance_tab()
    st.caption("Data: Jeff Sackmann's ATP match data (CC BY-NC-SA 4.0). Non-commercial portfolio project.")


main()
