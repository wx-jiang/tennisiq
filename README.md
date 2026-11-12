# TennisIQ: An LLM-Powered Tennis Prediction & Analytics Agent

TennisIQ predicts ATP match outcomes with an XGBoost model and an Elo baseline, both evaluated on a held-out season. It also includes an LLM analyst that answers natural-language questions by calling Python tools, so every number it states comes from the model or the match records rather than the LLM's memory.

> **Live demo:** _not yet deployed (see [Deployment](#deployment))._ · **Data cutoff:** 2025-12-17

<!-- Screenshot placeholders: add docs/prediction.png and docs/chat.png after deploying -->

## Key Results

All numbers below were measured by the scripts in `evaluation/` and are saved in `artifacts/`.

**Match prediction: held-out 2025 season (2,840 ATP tour-level matches, 2025-01-06 to 2025-12-17)**

| Model | Accuracy | Log loss | Brier | ROC-AUC |
|---|---|---|---|---|
| Elo baseline (overall + surface blend) | 0.630 | 0.638 | 0.223 | 0.694 |
| **XGBoost** | **0.657** | **0.620** | **0.216** | **0.712** |
| _Reference: higher-ranked player wins_ | _0.648_ | – | – | – |

- XGBoost reduces log loss vs. Elo by **0.017** (paired bootstrap 95% CI: 0.009 to 0.025, 2,000 resamples over matches), so the improvement is statistically distinguishable from zero.
- Elo on its own is slightly *less* accurate than simply picking the higher-ranked player (63.0% vs. 64.8%). A tour-level-only Elo starts every newcomer at 1500, which overrates qualifiers and lucky losers. XGBoost gains mainly by combining Elo with ATP rank/points.

**Agent benchmark (30 queries, same agent code and prompt, two LLMs)**

| Metric | `qwen3:14b` (local, Ollama) | `gemini-3.1-flash-lite` (cloud API) |
|---|---|---|
| Tool selection accuracy | 30/30 (100%) | 30/30 (100%) |
| Argument accuracy | 100% | 100% |
| Fact recall (ground-truth numbers present in answer) | 33/34 (97.1%) | 34/34 (100%) |
| Answers containing a number not found in any tool output | 0/30 | 0/30 |
| **End-to-end success** | **29/30 (96.7%)** | **30/30 (100%)** |
| Latency per query, median / p90 | 11.7 s / 15.0 s (M3 Pro, 36 GB) | 1.8 s / 3.2 s |

- **Qwen's one failure:** asked for Federer's 2017 record, it gave 54 wins in 59 matches but didn't state the losses, which the strict fact check requires.
- **Gemini latency:** the run was paced (`--pause 15`) to stay under the free tier's rate limit. One query took 28.4 s, consistent with a rate-limit retry, which raises the mean to 3.3 s.
- Each model was run once at temperature 0. A perfect score on 30 author-written queries shows the tools and validation work reliably. It doesn't mean the agent never makes mistakes.

## Features

- **Match prediction:** pick two players and a surface to get XGBoost and Elo win probabilities, Elo ratings, and the model inputs.
- **AI Tennis Analyst:** chat interface backed by a tool-calling agent (prediction, historical records, surface comparison, follow-up questions). Each answer shows the exact tool calls and raw outputs.
- **Model performance:** test metrics, bootstrap CI, calibration plot, and agent benchmark results.
- **Works without an LLM:** prediction and performance tabs need no API key; chat turns on when Ollama or a cloud API is configured.

## Architecture

```
                 ┌──────────────── offline pipeline ────────────────┐
 HF mirror of    │ src/data.py ──► src/features.py ──► src/train.py │──► artifacts/
 Sackmann CSVs ─►│ (download,      (chronological      (Elo baseline, │    xgb_model.json
                 │  clean)          Elo + form)        XGBoost, CV)   │    player_snapshot.csv
                 └───────────────────────────────────────────────────┘    player_stats.csv
                                                                           model_metrics.json
 Streamlit app.py ─────────────────────────────────────────────────────────────┐
   │ Match Prediction tab ──► src/tools.py ──► src/predict.py ──► artifacts    │
   │ AI Analyst tab ──► src/agent.py ──(≤5 iterations)──► src/llm.py           │
   │                       │  ▲                          ├─ Ollama (local)     │
   │                       ▼  │ JSON results             └─ OpenAI-compatible  │
   │                    src/tools.py (whitelist + argument validation)         │
   │ Model Performance tab ──► artifacts/*.json, calibration.png               │
   └───────────────────────────────────────────────────────────────────────────┘
```

| File | Role |
|---|---|
| `src/data.py` | Downloads pinned CSVs, filters surfaces/walkovers, sorts chronologically |
| `src/elo.py` | Elo with decaying K-factor (FiveThirtyEight constants) |
| `src/features.py` | Single chronological pass producing pre-match features; A/B orientation |
| `src/train.py` | Validation-based model selection, final fit, artifact export |
| `src/predict.py` | Player-name resolution, snapshot features, symmetrized prediction |
| `src/tools.py` | The 3 LLM tools, JSON schemas, validating dispatcher |
| `src/llm.py` | Provider layer: Ollama or any OpenAI-compatible API |
| `src/agent.py` | Bounded tool-calling loop with conversation history |
| `evaluation/` | `evaluate_models.py`, `evaluate_agent.py`, `test_queries.json` |

## Data

- **Source:** Jeff Sackmann's [`tennis_atp`](https://github.com/JeffSackmann/tennis_atp) ATP tour-level singles results (Grand Slams, Masters, ATP 250/500, Finals, Davis Cup, Olympics), licensed **CC BY-NC-SA 4.0**.
- **Availability note:** as of November 2026 the original GitHub repository returns 404. The pipeline downloads the same `atp_matches_YYYY.csv` files from a [Hugging Face mirror](https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive), **pinned to commit `8ac86f7c`** for reproducibility. (Sackmann's separate [Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject) is point-by-point data for a volunteer-charted subset of matches and does not have the complete results this model needs.)
- **Years:** 2015–2024 for training/validation, 2025 for the final test. 2010–2014 are also downloaded **only to warm up** Elo ratings and form features, so 2015 ratings aren't all 1500. No training rows come from them.
- **Cleaning:** keep Hard/Clay/Grass, drop walkovers/defaults (no information about playing strength). 45,098 matches in total, of which 30,346 are in 2015–2025.
- **Redistribution:** raw and processed match files are **not committed** (`data/` is git-ignored). The repo includes only derived artifacts (model, per-player feature snapshot, aggregated win/loss counts per player-year-surface), which are shared under the same CC BY-NC-SA 4.0 terms with attribution. This is a non-commercial project.

## Modeling Methodology

**Elo baseline.** Overall and surface-specific Elo, K = 250/(n+5)^0.4 (n = player's prior matches; constants from FiveThirtyEight, not tuned). The baseline probability uses a blend `w·overall + (1−w)·surface`. w ∈ {0, .25, .5, .75, 1} was chosen on validation log loss (w = 0.5).

**XGBoost.** Features are A-minus-B differences of each player's pre-match values:

| Feature | Definition |
|---|---|
| Elo, surface Elo | Ratings before the match |
| log rank, log rank points | ATP ranking at tournament start (unranked → rank 2000, 0 points) |
| Recent win rate | Last 20 matches, shrunk toward 0.5 with a Beta(2,2) prior |
| Surface win rate | Career (since 2010) on the match surface, same shrinkage |
| Experience | log(1 + matches), log(1 + matches on surface) |
| Age | Age at the match |
| Best-of-5 | Match format indicator |

**Validation.** Expanding-window chronological CV: train 2015→Y−1, validate on Y for Y ∈ {2022, 2023, 2024}. A small grid (depth ∈ {2,3,4} × trees ∈ {150,400}, lr 0.05) was scored by mean log loss. Grid differences were small (0.6064–0.6087). The best setting (depth 4, 400 trees) was refit on 2015–2024 (27,506 matches) and scored **once** on 2025.

**Orientation.** Each training match appears in both orientations (A = winner, y = 1 and A = loser, y = 0). At inference the prediction is averaged over both orientations, `p = ½[f(x) + 1 − f(−x)]`, so P(A beats B) + P(B beats A) = 1 exactly and the result can't depend on row order.

## Data Leakage Prevention

1. **Chronological single pass:** for each match, both players' features are read *before* the result updates their state. Matches are sorted by (tournament date, tournament, round), so results from later rounds of the same event can't leak backward. A unit test checks that features for the first *k* matches are identical whether or not later matches exist.
2. **No in-match statistics:** aces, serve points, score and duration are never used.
3. **Rankings** are the values published at tournament start, which are known before the match.
4. **Elo** is updated only after the match is recorded; no future outcomes are used.
5. **Test isolation:** Elo blend weight and XGBoost hyperparameters were chosen on 2022–2024 validation folds. 2025 is used only in `evaluate_models.py`.
6. **No ordering leak:** symmetric augmentation in training, orientation averaging at inference, and a seeded random orientation for AUC.

## Agent Design

- **Tools:** `predict_match(player1, player2, surface)`, `get_player_stats(player, year?, surface?)`, `compare_surfaces(player1, player2)`.
- **Loop:** at most 5 LLM calls per question. The model returns native tool calls, each call is validated (tool whitelist, schema keys, types, required fields), executed, and its JSON result is appended to the conversation. If the limit is reached, the agent returns a fixed fallback message.
- **Safety:** only the three whitelisted Python functions can run. No code or shell execution. Invalid players, surfaces or years come back as `{"error": ...}` for the LLM to explain or retry.
- **Grounding:** the system prompt requires every number to come verbatim from a tool result. Tools label outputs as *model estimates* or *historical facts* and include the model version, data cutoff, and staleness warnings (e.g. retired players).
- **Follow-ups:** conversation history (including tool results) is passed back on each turn, so "what if they played on grass?" reuses the earlier players.
- **Name resolution:** accent/case-insensitive, unique-surname matching ("Alcaraz"). Ambiguous surnames resolve only when one player has far more matches; otherwise the tool returns the candidates.

### Agent evaluation criteria (`evaluation/evaluate_agent.py`)

Scored programmatically against 30 queries (8 prediction, 8 historical stats, 5 surface comparison, 5 multi-step including 2 follow-ups, 4 invalid/unsupported):

1. **Tool selection:** the set of tools called equals an acceptable plan's set.
2. **Argument correctness:** expected calls matched after canonicalizing player names (resolved to the same player, order-insensitive), surface and year.
3. **Fact recall:** ground-truth values (computed by calling the tool directly) appear in the answer (0.596, 59.6% or 60% all count).
4. **Grounding:** every number in the answer appears in a tool result or the question (years, dates and the constants 0–3, 5, 20 are exempt).
5. **End-to-end success:** all of the above, plus for invalid inputs no percentage in the answer.
6. **Latency:** wall-clock seconds per query (setup turns and `--pause` waits excluded).

Per-model outputs are in `evaluation/results/agent_eval_results_<model>.json` and `artifacts/agent_eval_summary_<model>.json`. Each run also overwrites `agent_eval_results.json` and `agent_eval_summary.json`, which the app's Model Performance tab displays.

To benchmark a cloud model, set the `LLM_*`/`OPENAI_*` variables (see [Deployment](#deployment)) and pace requests on rate-limited free tiers:

```bash
python -m evaluation.evaluate_agent --pause 15
```

## Installation

```bash
git clone <your-repo-url> tennisiq && cd tennisiq
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Reproduce everything (≈1 minute, excluding the agent benchmark):

```bash
python -m src.data                      # download pinned CSVs → data/matches.parquet
python -m src.train                     # CV model selection, final fit, artifacts/
python -m evaluation.evaluate_models    # 2025 test metrics + calibration plot
pytest -q                               # unit tests
streamlit run app.py
```

## Ollama Setup (local LLM)

```bash
brew install ollama && brew services start ollama
ollama pull qwen3:14b                   # ~9 GB; runs comfortably on 36 GB Apple Silicon
python -m evaluation.evaluate_agent     # 30-query agent benchmark
```

`qwen3:14b` runs with thinking disabled and temperature 0. Override with `OLLAMA_MODEL` (any Ollama model with tool support, e.g. `qwen3:8b` for lower latency).

## Deployment

The deployed app uses committed artifacts only, so it needs no data download, GPU, or Ollama.

1. Push this repository to GitHub (the `artifacts/` folder must be committed; `data/` is ignored).
2. On [share.streamlit.io](https://share.streamlit.io), create an app from the repo with main file `app.py`.
3. **Optional, to enable the public chat:** in *App settings → Secrets*, paste the contents of `.streamlit/secrets.toml.example` with a real key. The template is set up for Google Gemini (key from [Google AI Studio](https://aistudio.google.com)), which is the cloud model benchmarked above. Any OpenAI-compatible endpoint with tool calling also works (OpenAI, Groq, Together, OpenRouter via `OPENAI_BASE_URL`); `LLM_PROVIDER = "openai"` refers to that API format, not the vendor. To test locally, copy the template to the git-ignored `.streamlit/secrets.toml`; never put a real key in the tracked `.example` file. Cloud inference is **billed per token** by the provider. Without secrets, the chat tab shows setup instructions and the rest of the app works normally.

Credentials are read only from environment variables / Streamlit secrets and are never hard-coded.

## Limitations

- **Frozen snapshot:** predictions use player state as of the last match in the data (2025-12-17), not live form, injuries or current rankings. Players inactive for over a year get an explicit warning.
- **Tour-level only:** no Challenger/qualifying results, so Elo for newcomers is crude. That is a likely reason Elo underperforms the ranking heuristic.
- **Modest edge:** tennis outcomes are noisy. The model is ~1 percentage point more accurate than "higher rank wins" on 2025 and better calibrated/probabilistic than Elo, but it isn't a betting model.
- **Fixed format assumption:** the tools assume best-of-3. Head-to-head records, fatigue and tournament context aren't modeled.
- **Win/loss counts** include Davis Cup and Olympics and exclude walkovers, so they can differ slightly from official ATP records.
- **Agent benchmark** is small (30 queries) and written by the author. It measures tool-use reliability and grounding, not open-domain robustness.

## License & Attribution

Code: MIT. Match data © Jeff Sackmann / Tennis Abstract, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Derived artifacts in `artifacts/` are shared under the same license.
