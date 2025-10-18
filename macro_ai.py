#!/usr/bin/env python3
# with_whom/macro/macro_ai.py
"""
Macro AI Scenario Generator for Treasury PoC

Module Overview:
----------------
This module generates AI-driven macro-behavioral stress scenarios for treasury portfolios using Grok LLM via LangChain.
It aggregates local data (news, metrics, macro, state) into prompts, invokes the LLM to produce scenarios matching a strict schema,
and handles retries for validity. Designed for standalone testing or integration with orchestrator_phase1.py.

Key Features:
-------------
- Prompt Building: Constructs detailed prompts from context blocks (NEWS, MACRO_DATA, BASELINE_METRICS, etc.).
- Schema Enforcement: Validates shocks, explanations, and justifications; retries on failure.
- Data Loading: Loads JSONs from data/ folder with fallbacks.
- Standalone Output: Generates and saves scenarios to standalone_scenarios.json for testing.
- Extensibility: Supports custom macro_inputs, portfolio_intel, and summary for flexible integration.

Regulatory Alignment:
---------------------
- Basel III/ICAAP: Scenarios incorporate shocks (rates, FX, liquidity) and behaviors (outflow multipliers, delays) for stress testing.
- CRD IV Pillar 2: Justifications tie to macro/news drivers for auditability.

Dependencies:
-------------
- Python 3.8+: json, typing, datetime, pathlib (standard library).
- LangChain: SystemMessage, HumanMessage (pip install langchain).
- Custom: llm_factory.get_llm(), infra.json_utils.parse_strict_json.

Usage in Workflow:
------------------
- Standalone: python macro_ai.py → standalone_scenarios.json (3 candidates).
- Integration: scenarios = generate_ai_scenarios(macro_inputs, portfolio_intel, portfolio_summary).
- Inputs: macro_inputs (news/macro), portfolio_intel (cash_agg/HQLA), portfolio_summary (str from metrics).

Configuration Notes:
--------------------
- n_candidates: Number of scenarios (default 3).
- max_retries: Per-scenario retries (default 3) for schema/shock/justification validation.
- Schema: Strict JSON object with shocks, behaviors, explanation, justification_map.

Error Handling:
---------------
- Retries: On invalid JSON/empty shocks/missing justification.
- Fallbacks: Empty dict on missing local data.
- Raises: RuntimeError on max retries exhausted.

Author:
-------
FintekMinds TreasuryAI PoC — AI Scenario Generation Module
Version: 2025-10-12R5 (Schema-Enforced with Retries)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, List, Callable
from datetime import datetime
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage
from with_whom.llm_factory import get_llm
from with_whom.infra.json_utils import parse_strict_json

print("[macro_ai] loaded v2025-10-12R5")  # trace

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "standalone_scenarios.json"  # New: For standalone test results

# ----------------------------- SCHEMA ----------------------------- #
SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "description": {"type": "string"},
        "centers": {"type": "object"},
        "shocks": {
            "type": "object",
            "properties": {
                "rates_bps": {"type": "object"},
                "fx_shift_pct": {"type": "object"},
                "credit_spread_bps": {"type": "object"},
                "liquidity_haircut_mult": {"type": "object"},
                "payment_delay_days": {"type": "object"},
                "outflow_accelerate_days": {"type": "object"},
                "ccy_outflow_mult": {"type": "object"},
                "ccy_inflow_mult": {"type": "object"},
            },
        },
        "behaviors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "instrument_type": {"type": "string"},
                    "criteria": {"type": "object"},
                    "effect": {"type": "object"},
                    "scope": {"type": "object"},
                },
                "required": ["action", "instrument_type", "criteria", "effect"],
            },
        },
        "explanation": {"type": "string"},
        "justification_map": {
            "type": "object",
            "additionalProperties": {"type": "string"}
        },
        "severity_level": {"type": "string", "enum": ["medium", "high"]},
    },
}


# ----------------------------- Utilities ----------------------------- #
def _safe_json(obj: Any) -> str:
    """
    Safely serialize object to JSON string.

    Args:
        obj: Any serializable object.

    Returns:
        str: JSON string or str(obj) on failure.
    """
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)


def _has_any_shock(shocks: Dict[str, Any]) -> bool:
    """
    Check if shocks dict has any non-zero values.

    Args:
        shocks: Dict of shocks.

    Returns:
        bool: True if any non-zero shock.
    """
    if not shocks:
        return False
    for _, v in (shocks or {}).items():
        if isinstance(v, dict) and any(float(x or 0) != 0 for x in v.values()):
            return True
    return False


def _has_justification(data: Dict[str, Any]) -> bool:
    """
    Validate presence of explanation and justification_map.

    Args:
        data: Parsed scenario dict.

    Returns:
        bool: True if valid.
    """
    expl = data.get("explanation")
    just = data.get("justification_map")
    return bool(expl and isinstance(expl, str) and expl.strip()) and bool(isinstance(just, dict) and len(just) > 0)


def _coerce_news_item(it: Any) -> Dict[str, Any]:
    """
    Normalize news item to compact dict.

    Args:
        it: Raw news item.

    Returns:
        Dict: Normalized fields (title, source, etc.).
    """
    """Normalize a news item to a compact dict for the prompt."""
    if isinstance(it, dict):
        return {
            "title": it.get("title") or it.get("headline") or it.get("name"),
            "source": it.get("source"),
            "impact_level": it.get("impact_level"),
            "affected_markets": it.get("affected_markets"),
            "timestamp": it.get("timestamp") or it.get("published_at") or it.get("pubDate"),
            "summary": it.get("summary") or it.get("description") or it.get("snippet"),
        }
    return {"title": str(it)}


def _extract_news_items(news: Any, limit: int = 12) -> List[Dict[str, Any]]:
    """
    Normalize news to list of items, indexed for citation.

    Args:
        news: Raw news data (list/dict).
        limit: Max items (default: 12).

    Returns:
        List: Normalized items with _idx.
    """
    """
    Normalize arbitrary news.json shapes to a list of items.
    Accept keys: items, news_items, articles, entries, or nested under 'news'.
    """
    items: List[Any] = []

    if isinstance(news, list):
        items = news
    elif isinstance(news, dict):
        # direct lists
        for k in ("items", "news_items", "articles", "entries"):
            v = news.get(k)
            if isinstance(v, list):
                items = v
                break

        # nested under 'news'
        if not items and "news" in news:
            v = news.get("news")
            if isinstance(v, list):
                items = v
            elif isinstance(v, dict):
                for k in ("items", "news_items", "articles", "entries"):
                    vv = v.get(k)
                    if isinstance(vv, list):
                        items = vv
                        break

    if not isinstance(items, list):
        items = []

    coerced = [_coerce_news_item(it) for it in items]
    # number the items so the model can cite them in justification_map
    for idx, item in enumerate(coerced[:limit]):
        item["_idx"] = idx
    return coerced[:limit]


def _load_local_data(file_name: str) -> Dict[str, Any]:
    """
    Load JSON from data/ folder.

    Args:
        file_name: Filename (e.g., 'ai_news.json').

    Returns:
        Dict: Parsed data or {} on missing/error.
    """
    """
    Load local JSON from data/ (e.g., ai_news.json, baseline_metrics.json).
    Returns {} if missing.
    """
    path = DATA_DIR / file_name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


# ----------------------------- Prompt builder ----------------------------- #
def _build_prompt(
        macro_inputs: Dict[str, Any],
        portfolio_intel: Dict[str, Any],
        portfolio_summary: Optional[str],
        retry: int = 0,
) -> str:
    """
    Build LLM prompt for scenario generation.

    Args:
        macro_inputs: Dict with news/macro/baseline/state.
        portfolio_intel: Portfolio intel dict.
        portfolio_summary: Optional str summary.
        retry: Retry count for prompt adjustment.

    Returns:
        str: Formatted prompt.
    """
    """
    AI-driven scenario design. ***AI decides magnitudes*** based on supplied context.
    We DO NOT inject deterministic values. We expose context blocks (incl. news) and
    require the model to produce rationale.
    """
    sev_bump = f" (retry {retry + 1}: broaden exploration / tail drivers)" if retry > 0 else ""

    ctx_blocks: List[str] = []

    # News from ai_news.json or macro_inputs["news"]
    raw_news = macro_inputs.get("news") or _load_local_data("ai_news.json").get("news")
    if raw_news:
        top_items = _extract_news_items(raw_news, limit=12)
        if top_items:
            ctx_blocks.append("NEWS (top-items, numbered):\n" + _safe_json(top_items))

    # Macro data from macro_data.json or macro_inputs["macro_data"]
    macro_data = macro_inputs.get("macro_data") or _load_local_data("macro_data.json")
    if macro_data:
        ctx_blocks.append("MACRO_DATA (CPI, rates, GDP):\n" + _safe_json(macro_data))

    # Baseline metrics from baseline_metrics.json or macro_inputs["baseline_metrics"]
    baseline_metrics = macro_inputs.get("baseline_metrics") or _load_local_data("baseline_metrics.json").get(
        "baseline_metrics", {})
    if baseline_metrics:
        ctx_blocks.append("BASELINE_METRICS (LCR, VaR, survival):\n" + _safe_json(baseline_metrics))

    # Portfolio intel from portfolio_intel.json
    p_intel = portfolio_intel or _load_local_data("portfolio_intel.json")
    if p_intel:
        ctx_blocks.append("PORTFOLIO_INTEL (cash_agg, HQLA by level):\n" + _safe_json(p_intel))

    # State summary from state.json
    state = _load_local_data("state.json")
    if state:
        state_snip = {
            "hqla_total_usd": state.get("total_hqla_usd"),
            "fx_rates": {k: v for k, v in state.get("fx_rates", {}).items() if k in ["EURUSD", "GBPUSD"]},
            "positions_by_currency": state.get("positions_by_currency", {})
        }
        ctx_blocks.append("STATE_SUMMARY (HQLA, FX rates, positions):\n" + _safe_json(state_snip))

    if portfolio_summary:
        ctx_blocks.append("PORTFOLIO_SUMMARY:\n" + str(portfolio_summary))

    context_blob = "\n\n".join(ctx_blocks) if ctx_blocks else "No extra context."

    return f"""
You are an AI Chief Treasury Officer designing a *macro-behavioural stress scenario*
for a global bank. Your goal is to discover the most plausible combination of
market and liquidity shocks that would *materially worsen* the bank’s key metrics:
Liquidity Coverage Ratio (LCR), Survival Days, 10-day Outflows, and 1-day-99% VaR.{sev_bump}

Context (verbatim from upstream systems):
{context_blob}

STRICT REQUIREMENTS (no exceptions):
1) Return **valid JSON only** matching the schema below.
2) Include **non-zero** values in at least one shock field in 'shocks'.
3) Provide a concise high-level **'explanation'** of the scenario (2–6 sentences).
4) Provide a **'justification_map'** with keys for each non-empty shock field and **each behavior[i]**.
   - For example: "rates_bps", "fx_shift_pct", "credit_spread_bps", "liquidity_haircut_mult",
     "payment_delay_days", "outflow_accelerate_days", "ccy_outflow_mult", "ccy_inflow_mult",
     and "behavior[0]", "behavior[1]", ...
   - In each value, explain **why** that shock/behavior is appropriate **today** and **cite the
     relevant NEWS item indices** using the numbered list (e.g., "[0, 3, 5]") or **blocks** (e.g., "MACRO_DATA CPI 2.9%", "PORTFOLIO_INTEL USD retail 50M").
   - If a driver is portfolio-specific (e.g., Level-2 HQLA, funding mix), make that link explicit.
5) Do **not** invent sources; only reason from the NEWS, MACRO_DATA, BASELINE_METRICS, PORTFOLIO_INTEL, and STATE_SUMMARY provided.

STRICT REQUIREMENTS (no exceptions):
1) Return **valid JSON only** matching the schema below.
2) Include **non-zero** values in at least one shock field in 'shocks'.
3) Provide a concise high-level **'explanation'** of the scenario (2–6 sentences).
4) Provide a **'justification_map'** with keys for each non-empty shock field and **each behavior[i]**.
   - For example: "rates_bps", "fx_shift_pct", "credit_spread_bps", "liquidity_haircut_mult",
     "payment_delay_days", "outflow_accelerate_days", "ccy_outflow_mult", "ccy_inflow_mult",
     and "behavior[0]", "behavior[1]", ...
   - In each value, explain **why** that shock/behavior is appropriate **today** and **cite the
     relevant NEWS item indices** using the numbered list (e.g., "[0, 3, 5]") or **blocks** (e.g., "MACRO_DATA CPI 2.9%", "PORTFOLIO_INTEL USD retail 50M").
   - If a driver is portfolio-specific (e.g., Level-2 HQLA, funding mix), make that link explicit.
5) Do **not** invent sources; only reason from the NEWS, MACRO_DATA, BASELINE_METRICS, PORTFOLIO_INTEL, and STATE_SUMMARY provided.

Return JSON only. No markdown. No prose outside JSON.
JSON schema to follow (do not repeat it in your output):
{_safe_json(SCHEMA)}
"""


# ----------------------------- Public API ----------------------------- #
def generate_ai_scenarios(
        macro_inputs: Optional[Dict[str, Any]] = None,
        portfolio_intel: Optional[Dict[str, Any]] = None,
        portfolio_summary: Optional[str] = None,
        n_candidates: int = 3,
        max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    Generate AI scenarios using LLM.

    Args:
        macro_inputs: Optional dict with news/macro.
        portfolio_intel: Optional portfolio intel.
        portfolio_summary: Optional str summary.
        n_candidates: Number of scenarios (default 3).
        max_retries: Retries per scenario (default 3).

    Returns:
        List: Valid scenarios.
    """
    macro_inputs = macro_inputs or {}
    portfolio_intel = portfolio_intel or {}
    llm = get_llm()

    # Load local data if not provided
    ai_news = _load_local_data("ai_news.json")
    baseline_metrics = _load_local_data("baseline_metrics.json")
    macro_data = _load_local_data("macro_data.json")
    state = _load_local_data("state.json")

    # Merge into macro_inputs
    if ai_news:
        macro_inputs["news"] = ai_news
    if macro_data:
        macro_inputs["macro_data"] = macro_data
    if baseline_metrics:
        macro_inputs["baseline_metrics"] = baseline_metrics
    if state:
        macro_inputs["state"] = state

    scenarios: List[Dict[str, Any]] = []
    for i in range(max(1, n_candidates)):
        data = None
        for retry in range(max_retries):
            system = SystemMessage(content=(
                "You are an AI Treasury Scenario Designer. "
                "Answer with JSON ONLY that matches the schema. Always include non-zero shocks, "
                "an 'explanation', and a 'justification_map' with citations to numbered NEWS items or blocks."
            ))
            human = HumanMessage(content=_build_prompt(macro_inputs, portfolio_intel, portfolio_summary, retry))
            try:
                resp = llm.invoke([system, human])
                raw_json = resp.content
                data = parse_strict_json(raw_json)
                if not _has_any_shock(data.get("shocks", {})):
                    raise RuntimeError("empty shocks")
                if not _has_justification(data):
                    raise RuntimeError("missing explanation/justification")
            except Exception:
                data = None
            if data:
                break
        if data is None:
            raise RuntimeError("LLM failed to produce usable scenario after retries")

        data["id"] = data.get("id") or f"S{i + 1}"
        data.setdefault("provenance", {})["generated_at"] = datetime.utcnow().isoformat()
        scenarios.append(data)

    return scenarios


# Backward-compatible helper
def choose_max_stress_scenario(scenarios: List[Dict[str, Any]],
                               evaluate_fn: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
    """
    Select scenario with max stress score.

    Args:
        scenarios: List of scenarios.
        evaluate_fn: Function to score scenario.

    Returns:
        Dict: Best (max score) scenario or {}.
    """
    if not scenarios:
        return {}
    best, best_score = None, float("-inf")
    for s in scenarios:
        try:
            sc = float(evaluate_fn(s))
        except Exception:
            sc = float("-inf")
        if sc > best_score:
            best, best_score = s, sc
    return best or {}


# ----------------------------- Standalone Test ----------------------------- #
def main():
    """
    Standalone test: Generate and save scenarios.
    """
    """Standalone test: Load data from data/ folder, generate scenarios, and save to JSON."""
    macro_inputs = {}
    portfolio_intel = _load_local_data("portfolio_intel.json")
    baseline_metrics = _load_local_data("baseline_metrics.json").get("baseline_metrics", {})

    # Construct portfolio_summary from baseline_metrics
    portfolio_summary = (
        f"LCR={baseline_metrics.get('LCR', 0.0):.2f} | "
        f"survival_days={baseline_metrics.get('survival_days', 0.0):.2f} | "
        f"VaR_1d_99={baseline_metrics.get('VaR_1d_99', 0.0):,.0f} | "
        f"worst_10bd_outflow={baseline_metrics.get('worst_10bd_outflow', 0.0):,.0f}"
    )

    # Generate scenarios
    scenarios = generate_ai_scenarios(macro_inputs, portfolio_intel, portfolio_summary, n_candidates=3)

    # Save to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(scenarios, f, indent=2, default=str)
    print(f"[macro_ai] Saved scenarios to {OUTPUT_FILE}")

    # Print results
    print("[macro_ai] Generated scenarios:")
    print(json.dumps(scenarios, indent=2, default=str))


if __name__ == "__main__":
    main()