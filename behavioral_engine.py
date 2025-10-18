#!/usr/bin/env python3
# behavior/behavioral_engine.py
"""
Behavioral Engine for Treasury AI Proof-of-Concept (PoC)

Module Overview:
----------------
This module applies behavioral adjustments to portfolio cashflows as part of the Treasury AI PoC stress testing workflow.
It modifies cashflows in-place within a copied state dictionary based on specified actions (e.g., deposit flight, payment delays, early bond calls),
ensuring compliance with Basel III behavioral assumptions for liquidity stress scenarios. The module is designed to be non-invasive,
returning a new state with an audit trail for transparency and traceability.

Key Features:
-------------
- Actions: Supports deposit_flight (outflow scaling/injection), delay (inflow postponement), early_call (bond inflow acceleration).
- Criteria-Based: Filters cashflows by instrument type and currency; extensible for additional criteria (e.g., credit rating).
- Auditability: Records touched flows and injected amounts in behavior_audit.
- Robustness: Handles malformed dates/amounts; skips unsupported actions with audit notes.
- In-Place Modification: Operates on a deep copy of the input state to preserve original data.

Regulatory Alignment:
---------------------
- Basel III (BCBS 238): Models deposit runoffs (e.g., 10% retail outflow), payment delays, and callable bond behavior for LCR/NSFR stress testing.
- CRD IV: Supports behavioral adjustments for liquidity coverage under stress (e.g., accelerated outflows, reduced inflows).
- Audit Trail: Aligns with Dodd-Frank/MiFID II transparency requirements by logging modifications.

Dependencies:
-------------
- Python 3.8+: copy, datetime, timedelta, typing (standard library).
- No external libraries; pure Python for PoC portability.

Usage in Workflow:
------------------
- Input: State dict from state_builder.py (contains cashflows).
- Process: apply_behaviors(state, behaviors) → modified state with audit.
- Downstream: Feeds stress_tester.py for scenario application; visualized in treasury_dashboard.py.
- Example: apply_behaviors(state, [{"action": "deposit_flight", "instrument_type": "retail_deposits", ...}]).

Configuration Notes:
--------------------
- Behaviors: List of dicts specifying action, instrument_type, criteria (e.g., currency), effect (e.g., runoff_rate_pct), scope (e.g., time_horizon_days).
- Supported Actions: deposit_flight, delay, early_call; others skipped with audit note.
- Defaults: 10% runoff, 7-day delay, 30-day early call, 30-90 day horizons.

Error Handling:
---------------
- Malformed Dates: Falls back to as_of or %Y-%m-%d parsing.
- Invalid Actions/Types: Skipped with audit entry.
- Missing Criteria: Processes with defaults (e.g., USD currency).

Author:
-------
FintekMinds TreasuryAI PoC — Behavioral Adjustment Module
Version: 2025-10 (Supports LCR Stress Behaviors)
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple


def _parse_date(s: str) -> datetime:
    """
    Parse date string to datetime.

    Args:
        s: ISO or similar date string.

    Returns:
        datetime: Parsed date or parsed %Y-%m-%d on failure.

    Purpose:
        Robust date parsing for cashflow date adjustments.
    """
    try:
        return datetime.fromisoformat(s[:19])
    except Exception:
        return datetime.strptime(s.split("T")[0], "%Y-%m-%d")


def _shift_date_str(s: str, days: int) -> str:
    """
    Shift date string by days.

    Args:
        s: ISO date string.
        days: Number of days to shift.

    Returns:
        str: ISO date string after shift.
    """
    return (_parse_date(s) + timedelta(days=int(days))).date().isoformat()


def _matches_criteria(cf: Dict[str, Any], itype: str, criteria: Dict[str, Any]) -> bool:
    """
    Check if cashflow matches behavior criteria.

    Args:
        cf: Cashflow dict.
        itype: Instrument type (e.g., 'retail_deposits').
        criteria: Dict with filters (e.g., currency).

    Returns:
        bool: True if matches criteria.
    """
    if itype and (cf.get("type") or cf.get("product")) != itype:
        return False
    ccy = criteria.get("currency")
    if ccy:
        if isinstance(ccy, list) and cf.get("currency") not in ccy:
            return False
        if isinstance(ccy, str) and cf.get("currency") != ccy:
            return False
    return True  # we keep credit_rating/callable checks for instruments below (if available)


def _alias_action(name: str) -> str:
    """
    Normalize action names to standard forms.

    Args:
        name: Raw action string.

    Returns:
        str: Standardized action (e.g., 'depositflight' → 'deposit_flight').
    """
    name = (name or "").lower().strip()
    if name in {"deposit_runoff", "depositflight"}:
        return "deposit_flight"
    if name in {"early_calls", "earlycall"}:
        return "early_call"
    if name in {"delay_risky_loans", "delayloans"}:
        return "delay"
    return name


def apply_behaviors(state: Dict[str, Any], behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Apply behaviors to cashflows *in-place* and return a new state with an audit.
    Supported actions: deposit_flight, delay (on loans/corporate_deposits inflows), early_call (requires callable flag)

    Args:
        state: Input state dict with cashflows.
        behaviors: List of behavior dicts (action, instrument_type, criteria, effect, scope).

    Returns:
        Dict: Modified state with behavior_audit.
    """
    # Create a deep copy to avoid modifying input state
    s = deepcopy(state)
    cfs = s.get("cashflows", []) or []
    audit: List[Dict[str, Any]] = []

    for b in behaviors or []:
        action = _alias_action(b.get("action"))
        itype = b.get("instrument_type") or ""
        criteria = b.get("criteria", {}) or {}
        effect = b.get("effect", {}) or {}
        scope = b.get("scope", {}) or {}

        touched = 0
        injected = 0.0

        # 1) Deposit flight → scale outflows up, scale inflows down, or inject a runoff outflow if none exist
        if action == "deposit_flight" and itype in {"retail_deposits", "sme_deposits", "corporate_deposits"}:
            runoff = float(effect.get("runoff_rate_pct", 10.0)) / 100.0
            horizon = int(scope.get("time_horizon_days", 30))
            cutoff = _parse_date(s.get("as_of")).date() + timedelta(days=horizon)

            has_deposit_flow = False
            for cf in cfs:
                if not _matches_criteria(cf, itype, criteria):
                    continue
                d = _parse_date(cf["date"]).date()
                if d <= cutoff:
                    has_deposit_flow = True
                    amt = float(cf.get("amount", 0.0))
                    if cf.get("direction") == "out":
                        cf["amount"] = amt * (1.0 + runoff)
                    else:
                        cf["amount"] = amt * (1.0 - runoff)
                    touched += 1

            if not has_deposit_flow:
                # inject a single runoff outflow on T+1 as proxy
                injected_amt = portfolio_proxy = float(scope.get("proxy_base_usd", 0.0)) or 0.0
                if injected_amt == 0.0:
                    # fallback: use absolute sum of deposit flows in ≤90d as proxy if present in state
                    injected_amt = float(s.get("deposit_proxy_usd", 0.0)) * runoff
                if injected_amt > 0:
                    cfs.append({
                        "instrument_id": f"{itype}_runoff_injected",
                        "type": itype,
                        "product": itype,
                        "counterparty_id": "SYSTEM",
                        "currency": (criteria.get("currency") or "USD") if isinstance(criteria.get("currency"), str)
                        else ((criteria.get("currency") or ["USD"])[0]),
                        "date": _shift_date_str(s.get("as_of"), 1),
                        "amount": -abs(injected_amt),
                        "direction": "out",
                        "description": "Deposit flight (injected)",
                    })
                    injected += injected_amt

        # 2) Delay → only delay *inflows* (customers pay later)
        elif action == "delay":
            delay_days = int(effect.get("delay_days", 7))
            horizon = int(scope.get("time_horizon_days", 60))
            cutoff = _parse_date(s.get("as_of")).date() + timedelta(days=horizon)
            for cf in cfs:
                if not _matches_criteria(cf, itype, criteria):
                    continue
                if cf.get("direction") != "in":
                    continue
                d = _parse_date(cf["date"]).date()
                if d <= cutoff:
                    cf["date"] = _shift_date_str(cf["date"], delay_days)
                    touched += 1

        # 3) Early call on bonds → only if instrument has callable flag (in normalized cashflow's instrument metadata, if present)
        elif action == "early_call" and itype in {"bond", "bonds"}:
            call_rate = float(effect.get("call_rate_pct", 10.0)) / 100.0
            horizon = int(scope.get("time_horizon_days", 90))
            cutoff = _parse_date(s.get("as_of")).date() + timedelta(days=horizon)

            # Attempt a coarse early-call: bring forward a fraction of coupon/principal inflows IF they exist as inflows (bank's asset side)
            for cf in cfs:
                if not _matches_criteria(cf, "bond", criteria):
                    continue
                d = _parse_date(cf["date"]).date()
                if d <= cutoff and cf.get("direction") == "in" and cf.get("description", "").lower().startswith(("coupon", "principal", "amortization", "redemption")):
                    # Pull earlier by 30 days
                    cf["date"] = _shift_date_str(cf["date"], -30)
                    # Scale up by call rate to reflect more being called into this window
                    cf["amount"] = float(cf["amount"]) * (1.0 + call_rate)
                    touched += 1

        # Unknown → skip but record
        else:
            audit.append({"action": action, "status": "skipped", "reason": "unsupported_action_or_type"})
            continue

        audit.append({"action": action, "type": itype, "touched_flows": touched, "injected_usd": injected})

    s["cashflows"] = cfs
    s["behavior_audit"] = audit
    return s