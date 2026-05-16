"""
irrigation_llm_bridge.py
LLM bridge for Irrigation-Game-LLM (NetLogo 7 + py: extension).

Two decisions per tick, per agent:
  1. decide_contribution(agent_id, ...) -> int 0-10
  2. decide_extraction(agent_id, ...) -> int 0-available_water

Position mapping: turtle who == position index (0=most upstream, 4=most downstream)
Position labels: 0->A, 1->B, 2->C, 3->D, 4->E
"""

from __future__ import annotations
import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# .env loader (NetLogo py: doesn't inherit terminal env vars)
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv()

# ---------------------------------------------------------------------------
# calpg — irrigation infrastructure production function (from original model)
# ---------------------------------------------------------------------------

def _calpg(invest: float) -> int:
    """Total investment -> available water (0-100)."""
    if   invest < 10: return 0
    elif invest < 15: return 5
    elif invest < 20: return 20
    elif invest < 25: return 40
    elif invest < 30: return 60
    elif invest < 35: return 75
    elif invest < 40: return 85
    elif invest < 45: return 95
    else:             return 100

POSITION_LABELS = ["A", "B", "C", "D", "E"]
POSITION_DESCS  = ["most upstream", "second upstream", "middle", "second downstream", "most downstream"]

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_cfg: dict = {}
_run_id: str = ""

# Per-agent data
_agent_backends: list[str] = []
_agent_models:   list[str] = []
_agent_memories: list[list[str]] = [[] for _ in range(5)]   # contribution+extraction memory per agent
_clients: list = [None] * 5

# CSV writers
_decisions_file = None
_decisions_writer = None
_institution_file = None
_institution_writer = None

# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------

def _make_client(backend: str, model: str, ollama_base_url: str = "http://localhost:11434/v1"):
    if backend == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    elif backend == "openai":
        import openai
        return openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    elif backend == "ollama":
        import openai
        return openai.OpenAI(base_url=ollama_base_url, api_key="ollama")
    elif backend == "google":
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        return genai.GenerativeModel(model)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

# ---------------------------------------------------------------------------
# LLM call (same multi-backend pattern as mastoc_llm_bridge)
# ---------------------------------------------------------------------------

def _call_llm(agent_id: int, system_prompt: str, user_prompt: str) -> tuple[str, str]:
    backend = _agent_backends[agent_id]
    model   = _agent_models[agent_id]
    client  = _clients[agent_id]
    max_tokens = _cfg.get("max_tokens", 256)

    if backend in ("anthropic",):
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text, ""

    elif backend in ("openai", "ollama"):
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content, ""

    elif backend == "google":
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        last_exc = None
        for attempt in range(4):
            try:
                resp = client.generate_content(
                    full_prompt,
                    generation_config={"max_output_tokens": max_tokens},
                )
                return resp.text, ""
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                is_rate_limit = (
                    "429" in err_str
                    or "quota" in err_str.lower()
                    or "ResourceExhausted" in type(exc).__name__
                )
                if is_rate_limit and attempt < 3:
                    m = re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', err_str)
                    wait = int(m.group(1)) + 2 if m else 15 * (attempt + 1)
                    print(f"[Google 429] Rate limit, waiting {wait}s (retry {attempt+1}/3)…")
                    time.sleep(wait)
                else:
                    raise
        raise last_exc

    else:
        raise ValueError(f"Unknown backend: {backend!r}")

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

PG_TABLE = (
    "Total investment → water available:\n"
    "  <10 → 0  | 10-14 → 5  | 15-19 → 20 | 20-24 → 40\n"
    "  25-29 → 60 | 30-34 → 75 | 35-39 → 85 | 40-44 → 95 | 45-50 → 100"
)

def _system_prompt(agent_id: int, override: str = "") -> str:
    if override.strip():
        return override.strip()
    pos_label = POSITION_LABELS[agent_id]
    pos_desc  = POSITION_DESCS[agent_id]
    return (
        f"You are an irrigation farmer at position {pos_label} ({pos_desc}) in a shared irrigation canal.\n"
        "Each round has two phases:\n"
        "  1. CONTRIBUTION: All farmers simultaneously invest 0-10 tokens in shared irrigation infrastructure.\n"
        "     The total investment determines how much water flows this round.\n"
        f"  {PG_TABLE}\n"
        "  2. EXTRACTION: Farmers extract water sequentially from upstream (A) to downstream (E).\n"
        "     You can only collect what remains after upstream farmers have taken their share.\n"
        "Your income each round = tokens you kept (10 - contribution) + water you collected.\n"
        "Act in your own long-term interest, but consider that cooperation benefits everyone."
    )

def _contribution_prompt(agent_id: int, tick: int, history: list[str]) -> str:
    pos_label = POSITION_LABELS[agent_id]
    mem = "\n".join(history[-_cfg.get("memory_length", 5):]) if history else "None yet."
    return (
        f"Round {tick + 1}. You are Farmer {pos_label}.\n"
        "Phase 1 — CONTRIBUTION.\n"
        "You have 10 tokens. How many do you invest in shared infrastructure (0–10)?\n"
        f"Your recent history:\n{mem}\n\n"
        "Reply with exactly:\n"
        "CONTRIBUTE: <integer 0-10>\n"
        "REASON: <one sentence> (optional)"
    )

def _extraction_prompt(agent_id: int, tick: int, contribution: int,
                        total_invest: float, pg: int, available: float,
                        already_extracted: float, history: list[str]) -> str:
    pos_label = POSITION_LABELS[agent_id]
    mem = "\n".join(history[-_cfg.get("memory_length", 5):]) if history else "None yet."
    return (
        f"Round {tick + 1}. You are Farmer {pos_label}.\n"
        "Phase 2 — EXTRACTION.\n"
        f"You invested {contribution} tokens this round.\n"
        f"Total group investment: {total_invest:.0f} → Infrastructure level (water): {pg}\n"
        f"Farmers upstream of you already extracted {already_extracted:.0f} units.\n"
        f"Water still available to you: {available:.0f}\n\n"
        f"Your recent history:\n{mem}\n\n"
        "How much water do you collect? You can take 0 to {available:.0f}.\n"
        "Reply with exactly:\n"
        "COLLECT: <integer 0-{available:.0f}>\n"
        "REASON: <one sentence> (optional)"
    ).format(available=int(available))

# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _parse_contribution(text: str) -> tuple[int, str]:
    m = re.search(r'CONTRIBUTE\s*:\s*(\d+)', text, re.IGNORECASE)
    val = int(m.group(1)) if m else 5  # default: moderate contribution
    val = max(0, min(10, val))
    reason_m = re.search(r'REASON\s*:\s*(.+)', text, re.IGNORECASE)
    reason = reason_m.group(1).strip() if reason_m else ""
    return val, reason

def _parse_extraction(text: str, available: float) -> tuple[int, str]:
    m = re.search(r'COLLECT\s*:\s*(\d+)', text, re.IGNORECASE)
    val = int(m.group(1)) if m else int(available / 2)  # default: half
    val = max(0, min(int(available), val))
    reason_m = re.search(r'REASON\s*:\s*(.+)', text, re.IGNORECASE)
    reason = reason_m.group(1).strip() if reason_m else ""
    return val, reason

# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

def _open_logs():
    global _decisions_file, _decisions_writer, _institution_file, _institution_writer
    log_dir = Path(_cfg["log_dir"]) / _run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    _decisions_file = open(log_dir / "decisions.csv", "w", newline="", encoding="utf-8")
    _decisions_writer = csv.writer(_decisions_file)
    _decisions_writer.writerow([
        "tick", "agent_id", "position", "backend", "model", "phase",
        "contribution", "available_water", "collected",
        "total_invest", "pg", "income",
        "reason", "raw_response"
    ])

    _institution_file = open(log_dir / "institution_scores.csv", "w", newline="", encoding="utf-8")
    _institution_writer = csv.writer(_institution_file)
    _institution_writer.writerow(["tick", "score", "summary"])

def _log_contribution(tick: int, agent_id: int, contribution: int,
                       reason: str, raw: str):
    if _decisions_writer is None:
        return
    _decisions_writer.writerow([
        tick, agent_id, POSITION_LABELS[agent_id],
        _agent_backends[agent_id], _agent_models[agent_id],
        "contribution",
        contribution, "", "",
        "", "", "",
        reason, raw[:500]
    ])
    _decisions_file.flush()

def _log_extraction(tick: int, agent_id: int, contribution: int,
                     available: float, collected: int,
                     total_invest: float, pg: int, income: float,
                     reason: str, raw: str):
    if _decisions_writer is None:
        return
    _decisions_writer.writerow([
        tick, agent_id, POSITION_LABELS[agent_id],
        _agent_backends[agent_id], _agent_models[agent_id],
        "extraction",
        contribution, available, collected,
        total_invest, pg, income,
        reason, raw[:500]
    ])
    _decisions_file.flush()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure(
    run_id: str,
    log_dir: str,
    memory_length: int,
    max_tokens: int,
    institution_check_interval: int,
    backends: list,          # list of 5 backend strings
    models: list,            # list of 5 model strings
    ollama_base_url: str = "http://localhost:11434/v1",
    system_prompt_override: str = "",
) -> str:
    global _cfg, _run_id, _agent_backends, _agent_models, _clients, _agent_memories

    _run_id = run_id
    _cfg = {
        "log_dir": log_dir,
        "memory_length": memory_length,
        "max_tokens": max_tokens,
        "institution_check_interval": institution_check_interval,
        "system_prompt_override": system_prompt_override,
    }

    _agent_backends = list(backends)
    _agent_models   = list(models)
    _agent_memories = [[] for _ in range(5)]
    _clients = [
        _make_client(backends[i], models[i], ollama_base_url)
        for i in range(5)
    ]

    _open_logs()

    # write run_meta.json
    meta_path = Path(log_dir) / run_id / "run_meta.json"
    meta = {
        "run_id": run_id,
        "condition": "full-llm",
        "agent_configs": [
            {"position": POSITION_LABELS[i], "backend": backends[i], "model": models[i]}
            for i in range(5)
        ],
        "memory_length": memory_length,
        "started_at": datetime.now().isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return run_id


def log_params(params: dict) -> None:
    """Called after setup sliders are read; stores experiment params in run_meta.json."""
    meta_path = Path(_cfg["log_dir"]) / _run_id / "run_meta.json"
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["experiment_params"] = {str(k): v for k, v in params.items()}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as exc:
        print(f"[log_params] Warning: {exc}")


def decide_contribution(
    agent_id: int,
    tick: int,
    system_override: str = "",
) -> int:
    """Phase 1: returns integer contribution 0-10."""
    sys_prompt  = _system_prompt(agent_id, system_override)
    user_prompt = _contribution_prompt(agent_id, tick, _agent_memories[agent_id])
    raw, _ = _call_llm(agent_id, sys_prompt, user_prompt)
    val, reason = _parse_contribution(raw)
    _log_contribution(tick, agent_id, val, reason, raw)
    # Store partial memory entry — will be completed after extraction
    _agent_memories[agent_id].append(f"Round {tick+1} | Contributed: {val}")
    return val


def decide_extraction(
    agent_id: int,
    tick: int,
    contribution: int,
    total_invest: float,
    pg: int,
    available_water: float,
    already_extracted: float,
    system_override: str = "",
) -> int:
    """Phase 2: returns integer extraction 0-available_water."""
    sys_prompt  = _system_prompt(agent_id, system_override)
    user_prompt = _extraction_prompt(
        agent_id, tick, contribution, total_invest, pg,
        available_water, already_extracted, _agent_memories[agent_id]
    )
    raw, _ = _call_llm(agent_id, sys_prompt, user_prompt)
    val, reason = _parse_extraction(raw, available_water)

    income = (10 - contribution) + val
    _log_extraction(tick, agent_id, contribution, available_water, val,
                    total_invest, pg, income, reason, raw)

    # Update memory with extraction outcome
    if _agent_memories[agent_id]:
        last = _agent_memories[agent_id][-1]
        _agent_memories[agent_id][-1] = (
            last + f" | Collected: {val}/{int(available_water)} | Income: {income:.0f}"
        )
    return val


def score_institution(tick: int, contribution_history: list, extraction_history: list,
                       system_override: str = "") -> int:
    """
    Secondary LLM pass every N ticks.
    Scores 0-10 how well the group has developed Ostrom-like norms.
    Uses agent 0's client (cheapest / first).
    """
    interval = _cfg.get("institution_check_interval", 5)
    if tick % interval != 0 or tick == 0:
        return -1

    summary = (
        f"Irrigation game — tick {tick}.\n"
        f"Recent contributions per agent (A-E): {contribution_history}\n"
        f"Recent extractions per agent (A-E): {extraction_history}\n"
    )
    sys_prompt = (
        "You assess collective action in an irrigation commons. "
        "Score 0-10 how well this group has developed stable cooperation norms "
        "(consistent contributions, fair extraction, no free-riding or upstream dominance). "
        "Reply with: SCORE: <0-10>\nSUMMARY: <one sentence>"
    )
    raw, _ = _call_llm(0, sys_prompt, summary)
    m = re.search(r'SCORE\s*:\s*(\d+)', raw, re.IGNORECASE)
    score = int(m.group(1)) if m else -1
    score = max(0, min(10, score))
    sm = re.search(r'SUMMARY\s*:\s*(.+)', raw, re.IGNORECASE)
    summary_text = sm.group(1).strip() if sm else raw[:200]

    if _institution_writer:
        _institution_writer.writerow([tick, score, summary_text])
        _institution_file.flush()

    return score


def shutdown() -> None:
    """Call from NetLogo when simulation ends."""
    if _decisions_file:
        _decisions_file.close()
    if _institution_file:
        _institution_file.close()
