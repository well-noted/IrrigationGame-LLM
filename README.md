# Irrigation-Game-LLM

An LLM-augmented extension of Janssen (2012) *An Agent-based Model Based on Field Experiments*, adapted from the [CoMSES model library](https://www.comses.net/codebases/3073/releases/1.0.0/).

Part of a research program extending classic agent-based commons models with large language model agents — see also [MASTOC-LLM](../MASTOC-LLM/).

<img width="295" height="286" alt="image" src="https://github.com/user-attachments/assets/930c11d4-3873-4d3c-8aba-35a53ad95536" />


---

## The original model

Janssen (2012) formalised irrigation commons experiments conducted in Colombia and Thailand (32 groups, 5 players each). The game captures the asymmetric nature of irrigation systems:

- **5 farmers** occupy positions A (upstream) through E (downstream)
- Each round has two phases: **contribution** to shared infrastructure, then **sequential extraction** of water upstream-to-downstream
- The original model uses Fehr-Schmidt inequality-aversion utility with experiential learning (parameters: α guilt, β altruism, λ sensitivity, γ₁/γ₂ learning rates)

The key empirical finding: upstream farmers ("stationary bandits") systematically extract more, leaving downstream farmers with less — even when total infrastructure investment is high.

---

## This extension

All five agents are replaced with LLM agents that reason in natural language about contribution and extraction decisions. The Fehr-Schmidt utility math is removed; what remains is the game structure, the nonlinear production function, and a two-phase prompt cycle.

### What stays the same
- 5-agent structure, positions A–E
- Sequential upstream→downstream extraction order
- `calpg` production function (nonlinear step: total invest → water available 0–100)
- 10-round experimental design (configurable)

### What changes
- Decision logic replaced by LLM calls (`decide_contribution`, `decide_extraction`)
- All Fehr-Schmidt/learning parameters removed
- Per-agent backend and model selection (Anthropic, OpenAI, Ollama, Google)
- Comprehensive CSV logging of both decisions per tick
- Ostrom institution detector (secondary LLM pass every N ticks)

---

## Game mechanics

### Phase 1 — Contribution (simultaneous)

All five farmers simultaneously choose how many tokens (0–10) to invest in shared irrigation infrastructure. Their kept tokens are `10 − contribution`.

Total investment determines water flow via a step function:

| Total invest | Water (pg) |
|:---:|:---:|
| < 10 | 0 |
| 10–14 | 5 |
| 15–19 | 20 |
| 20–24 | 40 |
| 25–29 | 60 |
| 30–34 | 75 |
| 35–39 | 85 |
| 40–44 | 95 |
| ≥ 45 | 100 |

### Phase 2 — Extraction (sequential A→E)

Starting with farmer A, each agent sees how much water remains and decides how much to collect. Income = `(10 − contribution) + collected`.

This creates a fundamental asymmetry: upstream agents face no scarcity risk; downstream agents receive only what is left.

---

## LLM integration

Each agent has its own system prompt establishing their position and role. The bridge (`irrigation_llm_bridge.py`) makes two LLM calls per agent per tick:

1. **`decide_contribution(agent_id, tick, system_override)`** → int 0–10  
   Agents reason about the production function threshold and expected peer behaviour.

2. **`decide_extraction(agent_id, tick, contribution, total_invest, pg, available, already_extracted, system_override)`** → int 0–available  
   Agents see how much water remains after upstream extraction and decide how much to take.

A third call, **`score_institution(tick, ...)`**, runs every N ticks to score collective norm development on a 0–10 Ostrom scale.

All decisions, reasoning, and outcomes are logged to `logs/<run_id>/decisions.csv`.

---

## Research questions

Relative to the original model, LLM agents open several new questions:

1. **Does position framing override cooperative training?** Will agents prompted as "upstream" extract more, as real humans do — or will LLM cooperative defaults flatten the upstream advantage?
2. **Can agents negotiate fair sharing rules?** The message field in each extraction prompt can carry prior-round reasoning; do norm-like messages emerge?
3. **Model size vs. position reasoning:** Smaller models (e.g. Llama 3B) may produce cooperative-sounding messages but fail to reason about sequential scarcity — do they overshoot downstream?
4. **Cross-position comparison:** Same model at different positions — does the agent's upstream/downstream framing shift its extraction behaviour?

---

## Setup

### Requirements

```
pip install anthropic openai google-generativeai python-dotenv
```

### API keys

Create a `.env` file in this directory (the bridge loads it automatically):

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
```

### Running

1. Open `Irrigation-Game-LLM.nlogo` in NetLogo 7
2. Select backends and models for each of the 5 agents (A–E)
3. Click **setup**, then **go**

Results appear in `logs/<run_id>/`:
- `run_meta.json` — experiment parameters and agent config
- `decisions.csv` — every contribution and extraction decision with reasoning
- `institution_scores.csv` — Ostrom scores over time

---

## File structure

```
Irrigation-Game-LLM/
├── Irrigation-Game-LLM.nlogo   NetLogo 7 model
├── irrigation_llm_bridge.py    Python LLM bridge
├── config.json                 Default agent config
├── requirements.txt
├── .env                        API keys (create this; not committed)
└── logs/
    └── <run_id>/
        ├── run_meta.json
        ├── decisions.csv
        └── institution_scores.csv
```

---

## Citation

Original model:  
Janssen, M.A. (2012). An Agent-based Model based on Field Experiments. In A. Smaijgl & O. Barreteau (eds.), *Empirical Agent-based Modelling: Challenges and Solutions*. Springer.

CoMSES entry: https://www.comses.net/codebases/3073/releases/1.0.0/
