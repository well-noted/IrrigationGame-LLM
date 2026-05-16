# Irrigation-Game-LLM

An LLM-augmented extension of Janssen (2012) *An Agent-based Model Based on Field Experiments*, adapted from the [CoMSES model library](https://www.comses.net/codebases/3073/releases/1.0.0/).

Part of a research program extending classic agent-based commons models with large language model agents — see also [MASTOC-LLM](../MASTOC-LLM/).

<div align="center">
  <img width="295" height="286" alt="image" src="https://github.com/user-attachments/assets/930c11d4-3873-4d3c-8aba-35a53ad95536" />
</div>


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

## Preliminary results

> ⚠️ These are single 10-round runs from initial testing. Replications and longer runs are needed before conclusions can be drawn.

Two completed runs so far: one with Gemma 4 (local via Ollama) and one with Claude Sonnet 4.6. Both ran 10 rounds with default settings (memory-length = 5, institution-check-interval = 5).

### Run 1 — Gemma 4 (gemma4:e4b, all 5 agents)

Contributions locked at 5 tokens per agent per round across all 10 rounds — no adaptation whatsoever. Total investment was always 25, producing pg = 60 every round. Extraction followed a stable geometric cascade: A took ~30, B ~15, C ~7, D ~4, E ~2–4. One anomalous tick (round 4) saw A take all 60 units, leaving B–E with nothing, then revert to the same pattern.

| Approx. income per round | A | B | C | D | E |
|---|---|---|---|---|---|
| (rounds 0–9, typical) | 35 | 20 | 12 | 9 | 7 |

Reasoning fields were almost entirely empty — the model returned decisions without logging justifications. Institution score: **0/10** at tick 5 (no summary generated). The behavior looks more like a fixed heuristic (contribute 5, take half of what remains) than strategic reasoning. The upstream–downstream inequality is large and stable, but the infrastructure doesn't collapse because contributions also don't decline.

### Run 2 — Claude Sonnet 4.6 (all 5 agents)

This run produced the more theoretically interesting result. The game played out as a cascading defection — not of the commons itself, but of the shared infrastructure.

**Contribution and water trajectory:**

| Round | Total invest | pg (water) | A contrib | B contrib | C contrib | D contrib | E contrib |
|-------|-------------|------------|-----------|-----------|-----------|-----------|-----------|
| 0     | 31          | 75         | 5         | 7         | 5         | 7         | 7         |
| 1     | 31          | 75         | 5         | 7         | 5         | 7         | 7         |
| 2     | 27          | 60         | 5         | 7         | 5         | 5         | 5         |
| 3     | 20          | 40         | 5         | 7         | 3         | 3         | 2         |
| 4     | 13          | 5          | 6         | 7         | 0         | 0         | 0         |
| 5     | 8           | 0          | 3         | 5         | 0         | 0         | 0         |
| 6–8   | 8 / 8 / 3  | 0          | 5/5/3     | 3/3/0     | 0         | 0         | 0         |
| 9     | 0           | 0          | 0         | 0         | 0         | 0         | 0         |

**The mechanism — B's extraction strategy unravels cooperation:**

Agent A attempted to extract fairly throughout: "Taking 25 leaves 50 for the four downstream farmers — fair given my upstream advantage." But Agent B, also a cooperative contributor, extracted all remaining water after A on every round it could. By round 0, after A took 25, B took all 50 — leaving C, D, and E with zero water despite their contributions of 5–7 tokens each.

C, D, and E correctly diagnosed the situation within two rounds:

> **Round 2 – Agent E:** *"After two rounds of contributing 7 and collecting nothing, I'll reduce my contribution to 5."*

> **Round 3 – Agent D:** *"I've contributed 7, 7, and 5 tokens but collected 0 water each time. I need to rethink my strategy."*

> **Round 4 – Agent C:** *"I've collected zero water for four consecutive rounds... contributing only benefits others at my expense. CONTRIBUTE: 0."*

By round 4, C, D, and E had all dropped to zero contribution. Total investment fell to 13 — barely above the minimum threshold — producing only 5 units of water. By round 5 (total invest = 8), the group fell below the 10-token threshold and generated no water at all. From round 5 through round 9, contribution continued to fall and water production remained at zero.

By round 9: all five agents contributed 0. The stable Nash equilibrium — everyone keeps all 10 tokens, earns 10/round, produces no shared infrastructure — was reached in nine rounds.

**Income inequality at different stages:**

| Round | A income | B income | C income | D income | E income |
|-------|----------|----------|----------|----------|----------|
| 0     | 30       | 53       | 5        | 3        | 3        |
| 3     | 25       | 23       | 7        | 7        | 8        |
| 4     | 9        | 3        | 10       | 10       | 10       |
| 9     | 10       | 10       | 10       | 10       | 10       |

The perverse outcome: downstream agents C, D, and E ended up *better off* by defecting (10 tokens/round guaranteed) than they were when cooperating (3–5 tokens/round, because upstream agents extracted everything). Their defection was individually rational at every step.

Institution score: **2/10** at tick 5 — *"contributions are heavily skewed toward only two agents (A and B) while C, D, and E contribute nothing, and the complete absence of any extractions suggests the system is largely non-functional rather than cooperatively stable."*

### Contrast with MASTOC-LLM

The irrigation game produces a qualitatively different outcome from MASTOC, using the same underlying model (Claude Sonnet 4.6). In MASTOC, LLM agents spontaneously converged to equal, sustainable herds within 22 ticks and maintained them indefinitely. Here, the commons collapsed into zero-infrastructure equilibrium within 5 rounds.

The structural difference is the two-phase decision. In MASTOC, defection (adding a cow) is symmetric — any agent can do it, and any agent can observe and respond to it. In the irrigation game, defection is structurally upstream-favoring: upstream agents can extract all water during Phase 2 regardless of what downstream agents contributed in Phase 1. This severs the link between contribution and benefit for downstream agents, making their defection individually rational even when they start out cooperative.

The irrigation game may be a harder institutional problem than MASTOC — not because the agents reason less well, but because the position asymmetry creates a structural obstacle that cooperative reasoning alone cannot overcome. Ostrom (1992) noted that real irrigation institutions require specifically that downstream parties have monitoring and enforcement rights; without those, the upstream extraction advantage is decisive.

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
