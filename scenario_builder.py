#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
with_whom/scenario_builder.py — PoC scenario engine (full file)

Goals:
- Baseline metrics match show_baseline.py (no inflow cap; gross fallback when raw net <=0).
- Stressed metrics compute both:
    * regulatory cap-net view (75% inflow cap), and
    * comparable LCR on the same basis as baseline (gross vs net) for fair comparisons.
- Force shocked cashflows into portfolio_view -> state_builder so aggregator caches can't erase stress.
- Normalize behavior percentages (e.g., 20 => 20%).
- HQLA haircuts persist via liquidity.hqla_effective (state_builder now prefers amount_usd_effective).
- Adversity guardrail: ensure stressed comparable LCR <= baseline_LCR * (1 - margin).

Module Overview:
----------------
This module is the core scenario engine for the Treasury AI Proof-of-Concept (PoC). It processes standalone scenarios,
applies shocks and behaviors to portfolio cashflows, recomputes liquidity and risk indicators, and ensures stressed outcomes
are adversely impacted compared to the baseline. It integrates with other modules like state_builder and counterpart_aggregator
to rebuild states under stress, providing a full audit trail for transparency.

Key Features:
-------------
- Shock Application: Handles FX shifts, HQLA haircuts, liquidity shocks (multipliers, delays).
- Behavior Normalization: Applies behaviors with percentage scaling and type aliases.
- Indicator Computation: Dual LCR views (cap-net for regulation, comparable for fairness).
- Guardrail: Enforces adversity by scaling outflows if stressed LCR not worse.
- VaR Scaling: Simple heuristic to react VaR to shocks (disclosed coefficients).
- Audit: Logs all modifications and computations for traceability.

Regulatory Alignment:
---------------------
- Basel III LCR: 75% inflow cap in stressed; gross fallback for conservatism (BCBS 238).
- NSFR Survival Days: Computed from effective LCR * 30.
- ECB Multi-Currency: FX-aware outflows/inflows.
- Adversity: Ensures stress worsens metrics per ICAAP guidelines.

Dependencies:
-------------
- Python 3.8+: json, re, sys, inspect, copy, datetime, pathlib, typing (standard library).
- Custom Modules: portfolio_aggregator, state_builder (imported dynamically).

Usage in Workflow:
------------------
- Standalone: python scenario_builder.py → scenario_results.json.
- Inputs: standalone_scenarios.json (scenarios), bank_profile.json, counterparty_data.json, etc.
- Outputs: scenario_results.json (with indicators, diffs, audit); per-scenario audit JSONs.

Configuration Notes:
--------------------
- INFLOW_CAP_STRESS: 0.75 (Basel default).
- LEVEL2_MAX_SHARE/LEVEL2B_MAX_SHARE: 0.40/0.15 (HQLA caps).
- ADVERSITY_MARGIN: 0.05 (5% worse LCR threshold).
- FIELD_MAP: Maps cashflow fields for flexibility.
- BEHAVIOR_ALIASES: Normalizes behavior types.

Error Handling:
---------------
- Skips invalid scenarios with audit notes.
- Graceful fallbacks on missing data (e.g., default FX=1.0).
- Exits on schema errors (e.g., missing as_of).

Author:
-------
FintekMinds TreasuryAI PoC — Scenario Building Engine
Version: 2025-10 (Adversity Guardrail & VaR Scaling)
"""

from __future__ import annotations

import json
import re
import sys
import inspect
from copy import deepcopy
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# ---------- Paths ----------
# Define base directories for data and logs
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR  = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Dictionary of file paths for inputs and outputs
FILES = {
    "standalone_scenarios": DATA_DIR / "standalone_scenarios.json",
    "ai_news":              DATA_DIR / "ai_news.json",
    "bank_profile":         DATA_DIR / "bank_profile.json",
    "counterparty_data":    DATA_DIR / "counterparty_data.json",
    "exchange_rate":        DATA_DIR / "exchange_rate.json",
    "portfolio_view":       DATA_DIR / "portfolio_view.json",
    "portfolio_intel":      DATA_DIR / "portfolio_intel.json",
    "baseline_state":       DATA_DIR / "baseline.json",
    "baseline_metrics":     DATA_DIR / "baseline_metrics.json",
    "state_json":           DATA_DIR / "state.json",
}
OUT_MAIN = DATA_DIR / "scenario_results.json"

# ---------- Basel parameters ----------
# Basel III inflow cap under stress (75%)
INFLOW_CAP_STRESS = 0.75
# Maximum share of Level 2/2B HQLA (40%/15%)
LEVEL2_MAX_SHARE  = 0.40
LEVEL2B_MAX_SHARE = 0.15
# Guardrail margin: stressed LCR <= baseline * (1 - 5%)
ADVERSITY_MARGIN  = 0.05   # 5% worse than baseline LCR (comparable basis)

# ---------- Flow field map ----------
# Mapping for cashflow fields to handle schema variations
FIELD_MAP = {
    "cashflows_key": "cashflows",
    "flow_date": "date",
    "flow_amount_usd": None,
    "flow_amount_native": "amount",
    "flow_currency": "currency",
    "flow_direction": "direction",  # "in" / "out"
    "flow_type": "type",
    "flow_product": "product",
}

# ---------- Behavior aliases ----------
# Aliases for normalizing behavior instrument types
BEHAVIOR_ALIASES = {
    "deposit": {"retail_deposits", "corporate_deposits", "deposit"},
    "bond": {"bond", "ust_bond", "corporate_bond", "ust_bill"},
    "commercial_paper": {"commercial_paper", "cp"},
    "certificate_of_deposit":{"certificate_of_deposit", "cd"},
    "repo": {"repo"},
    "swap": {"swap", "cross_currency_swap", "interest_rate_swap"},
    "cds": {"cds", "credit_default_swap"},
    "fx_forward": {"fx_forward"},
}

# ---------- Utils ----------
def _read_json(p: Path, default=None):
    """
    Read JSON file with fallback to default.

    Args:
        p: Path to JSON file.
        default: Value on missing or parse error.

    Returns:
        Any: Parsed JSON or deep-copied default.
    """
    if not p.exists():
        return deepcopy(default)
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"[ERROR] Failed reading {p}: {e}", file=sys.stderr)
        sys.exit(1)

def _write_json(p: Path, obj: Any):
    """
    Write object to JSON file.

    Args:
        p: Output path.
        obj: Serializable object.
    """
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str))

def _to_date(s: str | date | datetime) -> date:
    """
    Coerce input to date object.

    Args:
        s: Str, date, or datetime.

    Returns:
        date: Date part or parsed date.
    """
    if isinstance(s, date) and not isinstance(s, datetime):
        return s
    if isinstance(s, datetime):
        return s.date()
    try:
        return datetime.fromisoformat(str(s).split("T")[0]).date()
    except Exception:
        return datetime.strptime(str(s).split("T")[0], "%Y-%m-%d").date()

def _parse_as_of() -> date:
    """
    Resolve as_of date from state.json or portfolio_view.json.

    Returns:
        date: Parsed as_of or default '2025-10-15'.
    """
    s = _read_json(FILES["state_json"], None)
    if isinstance(s, dict):
        as_of = s.get("as_of") or (s.get("state") or {}).get("as_of")
        if as_of:
            print(f"[SCHEMA] as_of resolved to {_to_date(as_of)}")
            return _to_date(as_of)
    pv = _read_json(FILES["portfolio_view"], {})
    if isinstance(pv, dict) and pv.get("as_of"):
        print(f"[SCHEMA] as_of resolved to {_to_date(pv['as_of'])}")
        return _to_date(pv["as_of"])
    print("[WARN] as_of not found; defaulting to 2025-10-15", file=sys.stderr)
    return _to_date("2025-10-15")

def _within_30d(as_of_d: date, d: date) -> bool:
    """
    Check if date is within 30 days from as_of.

    Args:
        as_of_d: Reference date.
        d: Date to check.

    Returns:
        bool: True if as_of < d <= as_of + 30d.
    """
    return as_of_d < d <= (as_of_d + timedelta(days=30))

def _sign_from_direction(flow: dict) -> int:
    """
    Determine flow sign from direction or amount.

    Args:
        flow: Cashflow dict.

    Returns:
        int: +1 for inflow, -1 for outflow.
    """
    v = str(flow.get(FIELD_MAP["flow_direction"], "")).lower()
    if v.startswith("in"):  return +1
    if v.startswith("out"): return -1
    try:
        amt = float(flow.get(FIELD_MAP["flow_amount_native"], 0.0))
        return +1 if amt >= 0 else -1
    except Exception:
        return +1

def _flow_type_matches(flow: dict, wanted: str) -> bool:
    """
    Check if flow type/product matches wanted type.

    Args:
        flow: Cashflow dict.
        wanted: Desired type.

    Returns:
        bool: True if matches (case-insensitive, aliases).
    """
    ft = str(flow.get(FIELD_MAP["flow_type"], "")).lower()
    fp = str(flow.get(FIELD_MAP["flow_product"], "")).lower()
    w = wanted.lower()
    if ft == w or fp == w:
        return True
    for canon, aliases in BEHAVIOR_ALIASES.items():
        if w == canon and (ft in aliases or fp in aliases):
            return True
    return False

def _norm_level(lvl: str) -> str:
    """
    Normalize HQLA level strings.

    Args:
        lvl: Raw level string.

    Returns:
        str: Standardized level (e.g., 'Level 1').
    """
    s = str(lvl).strip().lower().replace(" ", "")
    if s in {"level1", "l1"}: return "Level 1"
    if s in {"level2a", "l2a"}: return "Level 2A"
    if s in {"level2b", "l2b"}: return "Level 2B"
    if "level" in str(lvl).lower(): return str(lvl).replace("level", "Level").strip()
    return str(lvl)

# ---------- SCHEMA SELF-CHECK ----------
def schema_self_check():
    """
    Validate input schema for required structures.

    Raises:
        SystemExit: On schema errors.
    """
    errs = []
    raw_scen = _read_json(FILES["standalone_scenarios"], [])
    scenarios = raw_scen.get("scenarios", []) if isinstance(raw_scen, dict) else raw_scen
    if not isinstance(scenarios, list) or not scenarios:
        errs.append("standalone_scenarios.json must be a non-empty list (or dict with 'scenarios').")

    cp = _read_json(FILES["counterparty_data"], [])
    if not isinstance(cp, list) or not cp:
        errs.append("counterparty_data.json must be a non-empty list of counterparties.")
    else:
        cf_total = 0
        required = {"date","amount","currency","direction"}
        for i, c in enumerate(cp[:5]):
            cflows = c.get("cashflows")
            if not isinstance(cflows, list):
                errs.append(f"counterparty_data[{i}].cashflows must be a list.")
                continue
            cf_total += len(cflows)
            for f in cflows[:5]:
                miss = [k for k in required if k not in f]
                if miss:
                    errs.append(f"Sample cashflow missing fields: {miss}")
        print(f"[SCHEMA] Counterparties={len(cp)} | Sampled flows fields OK | Total flows (first few cps)≈{cf_total}")

    fx = _read_json(FILES["exchange_rate"], {})
    if "USD" not in fx:
        errs.append("exchange_rate.json must include 'USD': 1.0 base.")
    else:
        try:
            if abs(float(fx["USD"]) - 1.0) > 1e-6:
                errs.append("'USD' must be 1.0.")
        except Exception:
            errs.append("'USD' must be numeric 1.0.")

    bp = _read_json(FILES["bank_profile"], {})
    liq = bp.get("liquidity", {})
    hb = liq.get("hqla_breakdown", {})
    if not isinstance(hb, dict) or not hb:
        errs.append("bank_profile.liquidity.hqla_breakdown must be a non-empty dict.")

    try:
        _ = _parse_as_of()
    except Exception as e:
        errs.append(f"Cannot resolve as_of: {e}")

    if errs:
        for e in errs:
            print(f"[SCHEMA-ERROR] {e}", file=sys.stderr)
        print("[SCHEMA] ✖ Aborting due to schema errors above.", file=sys.stderr)
        sys.exit(1)
    else:
        print("[SCHEMA] ✔ Schema self-check passed.")

# ---------- HQLA haircuts (effective + caps) ----------
def apply_hqla_haircuts(profile: dict, haircut_mult: dict, audit: List[str]) -> dict:
    """
    Apply HQLA haircuts and Basel caps.

    Args:
        profile: Bank profile dict.
        haircut_mult: Dict of multipliers by level.
        audit: List for audit notes.

    Returns:
        Dict: Updated profile with hqla_effective.
    """
    prof = deepcopy(profile) if profile else {}
    liq  = prof.get("liquidity", prof)
    hb   = (liq.get("hqla_breakdown") or {}) if isinstance(liq.get("hqla_breakdown"), dict) else {}

    base_eff = {"Level 1": 0.0, "Level 2A": 0.0, "Level 2B": 0.0}
    for lvl_raw, node in hb.items():
        lvl = _norm_level(lvl_raw)
        try:
            amt = float((node or {}).get("amount_usd", node.get("amount", 0.0)) or 0.0)
            base_hc = float((node or {}).get("haircut", 0.0))
        except Exception:
            amt, base_hc = 0.0, 0.0
        if lvl in base_eff:
            base_eff[lvl] += max(0.0, amt * (1.0 - base_hc))

    stressed = dict(base_eff)
    for k, mult in (haircut_mult or {}).items():
        lvl = _norm_level(k)
        try:
            m = float(mult)
            if m <= 0:
                continue
        except Exception:
            continue
        if lvl in stressed:
            before = stressed[lvl]
            stressed[lvl] = before / m
            audit.append(f"HQLA haircut: {lvl} effective {before:,.2f} → {stressed[lvl]:,.2f} (÷{m:g})")
        else:
            audit.append(f"HQLA haircut: {lvl} not present; no-op")

    total_pre = sum(stressed.values())
    l1  = stressed.get("Level 1", 0.0)
    l2a = stressed.get("Level 2A", 0.0)
    l2b = stressed.get("Level 2B", 0.0)
    if total_pre > 0.0:
        cap_l2b = total_pre * LEVEL2B_MAX_SHARE
        if l2b > cap_l2b:
            audit.append(f"L2B cap: {l2b:,.2f} → {cap_l2b:,.2f}")
            l2b = cap_l2b
        l2 = l2a + l2b
        cap_l2 = total_pre * LEVEL2_MAX_SHARE
        if l2 > cap_l2:
            excess = l2 - cap_l2
            take = min(l2a, excess)
            l2a -= take
            excess -= take
            if excess > 0:
                l2b = max(l2b - excess, 0.0)
            audit.append("L2 cap enforced (≤40%).")

    liq["hqla_effective"] = {"Level 1": l1, "Level 2A": l2a, "Level 2B": l2b}
    prof["liquidity"] = liq
    audit.append(f"HQLA effective total: {l1 + l2a + l2b:,.2f}")
    return prof

# ---------- FX ----------
def apply_fx_shifts(fx_rates: dict, fx_shift_pct: dict, audit: List[str]) -> dict:
    """
    Apply FX shifts to rates.

    Args:
        fx_rates: Base FX rates dict.
        fx_shift_pct: Dict of percentage shifts by ccy/pair.
        audit: List for audit notes.

    Returns:
        Dict: Updated FX rates.
    """
    rates = deepcopy(fx_rates) if fx_rates else {}

    def _ccy_of(k: str) -> Optional[str]:
        """
        Extract currency from key (e.g., 'EUR/USD' → 'EUR').
        """
        s = str(k).upper().strip()
        if re.match(r"^[A-Z]{3}/USD$", s):   # EUR/USD
            return s[:3]
        if re.match(r"^USD/[A-Z]{3}$", s):   # USD/EUR
            return s[-3:]
        if re.match(r"^[A-Z]{6}$", s):       # EURUSD or USDEUR
            return s[:3] if s.endswith("USD") else s[-3:]
        if re.match(r"^[A-Z]{3}$", s):       # EUR
            return s
        return None

    for k, v in (fx_shift_pct or {}).items():
        ccy = _ccy_of(k)
        if not ccy:
            audit.append(f"FX shift {k}: unrecognized; no-op")
            continue
        if ccy not in rates:
            audit.append(f"FX shift {k}: {ccy} not in rates; no-op")
            continue
        before = float(rates[ccy])
        after  = before * (1.0 + float(v) / 100.0)
        rates[ccy] = after
        audit.append(f"FX shift {k}: {before:.6f} → {after:.6f} ({float(v):+g}%)")
    return rates

# ---------- Baseline-style LCR (validator parity) ----------
def compute_lcr_show_basis(state: dict, force_basis: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute LCR with basis (gross/net) for comparable deltas.

    Args:
        state: State dict with hqla and cashflows.
        force_basis: Optional 'gross' or 'net' to override.

    Returns:
        Dict: With hqla, inflows, outflows, lcr_gross/net/effective, basis.
    """
    as_of = datetime.fromisoformat(str(state.get("as_of"))[:10])
    end   = as_of + timedelta(days=30)

    # HQLA prefer effective
    hqla = 0.0
    for h in state.get("hqla", []):
        try:
            amt = h.get("amount_usd_effective")
            if amt is None:
                amt = h.get("amount_usd", h.get("amount", 0.0))
            hqla += float(amt or 0.0)
        except Exception:
            pass

    inflow = outflow = 0.0
    for cf in state.get("cashflows", []):
        try:
            dt_raw = cf.get("date") or cf.get("value_date") or as_of
            dt = datetime.fromisoformat(str(dt_raw)[:10])
        except Exception:
            dt = as_of
        if not (as_of <= dt <= end):
            continue
        amt = float(cf.get("amount") or 0.0)  # NO FX to mirror validator
        d   = str(cf.get("direction") or "").lower()
        if d in ("out","pay","debit","payout","-1","-") or amt < 0:
            outflow += abs(amt)
        else:
            inflow += abs(amt)

    raw_net = outflow - inflow
    net_clip = max(0.0, raw_net)
    lcr_gross = (hqla / outflow) if outflow > 0 else None
    lcr_net   = (hqla / net_clip) if net_clip > 0 else None

    if force_basis in {"gross","net"}:
        basis = force_basis
        eff = lcr_gross if basis == "gross" else lcr_net
        if eff is None and basis == "net":
            eff = lcr_gross
    else:
        basis = "gross" if raw_net <= 0 else "net"
        eff = lcr_gross if basis == "gross" else lcr_net

    return {
        "hqlatotal_usd": hqla,
        "inflows_30d": inflow,
        "outflows_30d": outflow,
        "raw_net_outflows": raw_net,
        "lcr_gross": lcr_gross,
        "lcr_net": lcr_net,
        "lcr_effective": eff,
        "basis": basis,
    }

# ---------- Indicators (baseline_show vs stressed) ----------
def compute_indicators_from_state(state: dict,
                                  as_of_d: date,
                                  fx_rates: dict,
                                  audit: List[str],
                                  mode: str = "stressed") -> Dict[str, float]:
    """
    Compute indicators from state (stressed or baseline_show).

    Args:
        state: State dict.
        as_of_d: As-of date.
        fx_rates: FX rates dict.
        audit: List for audit notes.
        mode: 'stressed' or 'baseline_show'.

    Returns:
        Dict: With liquidity and VaR indicators.
    """
    out = {
        "hqlatotal_usd": 0.0, "gross_outflows_30d_usd": 0.0, "inflows_30d_usd": 0.0,
        "net_outflows_30d_usd": 0.0, "lcr_net": None, "lcr_gross": None,
        "lcr_effective": None, "survival_days_effective": None, "var_1d_99_usd": None,
        "raw_net_outflows": None,
    }

    # HQLA (prefer effective)
    hqla_list = state.get("hqla") or []
    hqla_total = 0.0
    for h in hqla_list:
        try:
            amt = h.get("amount_usd_effective")
            if amt is None:
                amt = h.get("amount_usd", h.get("amount", 0.0))
            hqla_total += float(amt or 0.0)
        except Exception:
            pass
    out["hqlatotal_usd"] = hqla_total

    # VaR passthrough
    var = (state.get("profile_var") or {}).get("var_1d_99_usd")
    out["var_1d_99_usd"] = float(var) if var is not None else None

    if mode == "baseline_show":
        comp = compute_lcr_show_basis(state)
        out["inflows_30d_usd"]        = comp["inflows_30d"]
        out["gross_outflows_30d_usd"] = comp["outflows_30d"]
        out["raw_net_outflows"]       = comp["raw_net_outflows"]
        out["net_outflows_30d_usd"]   = max(0.0, comp["raw_net_outflows"])
        out["lcr_gross"]              = comp["lcr_gross"]
        out["lcr_net"]                = comp["lcr_net"]
        out["lcr_effective"]          = comp["lcr_effective"]
        out["survival_days_effective"]= (30.0 * comp["lcr_effective"]) if comp["lcr_effective"] else None
        audit.append(
            f"Baseline(show) HQLA={hqla_total:,.2f} | "
            f"GrossOut={out['gross_outflows_30d_usd']:,.2f} Inflows={out['inflows_30d_usd']:,.2f} | "
            f"RawNet={out['raw_net_outflows']:,.2f} → LCR*={(out['lcr_effective'] if out['lcr_effective'] else float('nan')):.6f}"
        )
        return out

    # Stressed (FX-aware + 75% inflow cap)
    flows = state.get("cashflows") or []
    gross_out, inflows = 0.0, 0.0
    horizon_end = as_of_d + timedelta(days=30)

    def _to_date_safe(s):
        try:
            return datetime.fromisoformat(str(s).split("T")[0]).date()
        except Exception:
            return None

    for f in flows:
        fd = _to_date_safe(f.get("date"))
        if not fd or not (as_of_d < fd <= horizon_end):
            continue
        try:
            amt_native = float(f.get("amount", 0.0))
        except Exception:
            continue
        ccy = (f.get("currency") or "USD").upper()
        rate = float(fx_rates.get(ccy, 1.0)) if ccy != "USD" else 1.0
        amt_usd = amt_native * rate
        sgn = _sign_from_direction(f)
        if sgn < 0: gross_out += abs(amt_usd)
        else:       inflows += abs(amt_usd)

    out["gross_outflows_30d_usd"] = gross_out
    out["inflows_30d_usd"]        = inflows
    capped_inflows = min(inflows, INFLOW_CAP_STRESS * gross_out)
    net_out = gross_out - capped_inflows
    out["net_outflows_30d_usd"]   = max(net_out, 0.0)

    lcr_net  = (hqla_total / out["net_outflows_30d_usd"]) if out["net_outflows_30d_usd"] > 0 else None
    lcr_gross= (hqla_total / gross_out) if gross_out > 0 else None
    eff      = lcr_net if lcr_net is not None else lcr_gross

    out["lcr_net"]       = lcr_net
    out["lcr_gross"]     = lcr_gross
    out["lcr_effective"] = eff
    out["survival_days_effective"] = (30.0 * eff) if eff else None

    audit.append(
        f"Stressed HQLA={hqla_total:,.2f} | "
        f"GrossOut={gross_out:,.2f} Inflows={inflows:,.2f} (cap={int(INFLOW_CAP_STRESS*100)}%) "
        f"NetOut={out['net_outflows_30d_usd']:,.2f} → LCR*={(eff if eff else float('nan')):.6f}"
    )
    return out

# ---------- Liquidity shocks ----------
def clamp_into_window(as_of_d: date, d: date) -> date:
    """
    Clamp date into [as_of+1d, as_of+30d].

    Args:
        as_of_d: Reference date.
        d: Date to clamp.

    Returns:
        date: Clamped date.
    """
    lo = as_of_d + timedelta(days=1)
    hi = as_of_d + timedelta(days=30)
    if d < lo: return lo
    if d > hi: return hi
    return d

# ---------- Liquidity shocks (robust) ----------
def _coerce_shock_map(raw, cast=float):
    """
    Coerce a shocks input into a mapping of UPPERCASE keys -> cast(value).
    Accepts:
      - dict like {'USD': 0.5, 'EUR': 0.6}
      - scalar like 3 or "3" -> becomes {'*': cast(3)}
      - None -> {}
    'cast' can be int or float depending on expected type.
    """
    out = {}
    if raw is None:
        return out
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                key = str(k).upper()
                out[key] = cast(v)
            except Exception:
                # ignore unparseable entries
                continue
        return out
    # scalar -> treat as global default
    try:
        out["*"] = cast(raw)
    except Exception:
        pass
    return out


def apply_liquidity_shocks_to_flows(flows: List[dict], shocks: dict, as_of_d: date, audit: List[str]) -> Tuple[List[dict], Dict[str, Any]]:
    """
    Apply liquidity shocks to flows.

    Behaviour (compatible with old API):
      - shocks may supply per-currency dicts OR a scalar (applies to all currencies).
      - when looking up currency-specific values, prefer exact currency key, otherwise use '*' (global).
      - keeps original clamping to as_of+1 .. as_of+30 window for date shifts.

    Returns:
      - modified flows list (in-place edits applied to provided list),
      - summary dict describing applied shocks.
    """
    summary = {"ccy_inflow_mult": {}, "ccy_outflow_mult": {}, "payment_delay_days": {}, "outflow_accelerate_days": {}}
    if not flows:
        return flows, summary

    # Coerce incoming shock specs safely (accept dicts or scalars)
    inflow_mult_map  = _coerce_shock_map(shocks.get("ccy_inflow_mult"), float)
    outflow_mult_map = _coerce_shock_map(shocks.get("ccy_outflow_mult"), float)
    pay_delay_map    = _coerce_shock_map(shocks.get("payment_delay_days"), int)
    out_accel_map    = _coerce_shock_map(shocks.get("outflow_accelerate_days"), int)

    touched_examples = []

    for f in flows:
        ccy = (f.get("currency") or "USD").upper()
        try:
            fd = _to_date(f.get("date"))
        except Exception:
            # skip if date cannot be parsed
            continue
        sgn = _sign_from_direction(f)

        # helper: lookup from map, prefer exact ccy, else global '*', else None
        def _lookup(m):
            if not m:
                return None
            return m.get(ccy) if ccy in m else m.get("*")

        # inflow multiplier
        if sgn > 0:
            mult = _lookup(inflow_mult_map)
            if mult is not None:
                try:
                    before = float(f.get("amount", 0.0))
                except Exception:
                    before = 0.0
                f["amount"] = before * float(mult)
                summary["ccy_inflow_mult"][ccy] = mult if ccy in inflow_mult_map else inflow_mult_map.get("*")
                if len(touched_examples) < 5:
                    touched_examples.append({"currency": ccy, "direction": "inflow", "amount_before": before, "after": f["amount"], "date": str(fd)})

        # outflow multiplier
        if sgn < 0:
            mult = _lookup(outflow_mult_map)
            if mult is not None:
                try:
                    before = float(f.get("amount", 0.0))
                except Exception:
                    before = 0.0
                f["amount"] = before * float(mult)
                summary["ccy_outflow_mult"][ccy] = mult if ccy in outflow_mult_map else outflow_mult_map.get("*")
                if len(touched_examples) < 5:
                    touched_examples.append({"currency": ccy, "direction": "outflow", "amount_before": before, "after": f["amount"], "date": str(fd)})

        # payment delay for inflows (use global default if provided)
        delay = _lookup(pay_delay_map)
        if sgn > 0 and delay is not None and int(delay) != 0:
            newd = clamp_into_window(as_of_d, fd + timedelta(days=int(delay)))
            if newd != fd:
                f["date"] = newd.isoformat()
                # record the effective value under the currency key used (exact or '*')
                summary["payment_delay_days"][ccy] = int(delay) if ccy in pay_delay_map else pay_delay_map.get("*")

        # outflow acceleration (move earlier) for outflows
        accel = _lookup(out_accel_map)
        if sgn < 0 and accel is not None and int(accel) != 0:
            newd = clamp_into_window(as_of_d, fd - timedelta(days=abs(int(accel))))
            if newd != fd:
                f["date"] = newd.isoformat()
                summary["outflow_accelerate_days"][ccy] = int(accel) if ccy in out_accel_map else out_accel_map.get("*")

    audit.append(f"Liquidity shocks applied: examples={touched_examples}")
    return flows, summary

# ---------- Behaviors (percent normalized) ----------
def apply_behaviors(flows: list, behaviors: list, as_of_d: date, audit: list):
    """
    Apply behaviors to flows with percentage normalization.

    Args:
        flows: List of cashflows.
        behaviors: List of behaviors.
        as_of_d: As-of date.
        audit: List for audit notes.

    Returns:
        Tuple: Modified flows, behavior summaries.
    """
    if not flows or not behaviors:
        return flows, []

    audits = []

    def _sign(f):
        v = str(f.get("direction", "")).lower()
        if v.startswith("in"): return +1
        if v.startswith("out"): return -1
        try:
            return +1 if float(f.get("amount", 0.0)) >= 0 else -1
        except Exception:
            return +1

    def _norm_action(a: str) -> str:
        a = (a or "").strip().lower()
        mapping = {
            "accelerate_outflow": "accelerate_outflows",
            "accelerate_outflows": "accelerate_outflows",
            "delay_inflow": "delay_inflows",
            "delay_inflows": "delay_inflows",
            "haircut_inflow": "haircut_inflows",
            "haircut_inflows": "haircut_inflows",
        }
        return mapping.get(a, a)

    def clamp(asof: date, d: date) -> date:
        lo, hi = asof + timedelta(days=1), asof + timedelta(days=30)
        if d < lo: return lo
        if d > hi: return hi
        return d

    for b in behaviors:
        action = _norm_action(b.get("action"))
        effect = b.get("effect") or {}
        crit   = b.get("criteria") or {}
        scope  = b.get("scope") or {}

        pct = float(scope.get("percentage", 1.0))
        if pct > 1.0:
            pct /= 100.0
        pct = max(0.0, min(1.0, pct))

        want_type = str(crit.get("instrument_type", "")).lower()
        want_ccy  = (str(crit.get("currency", "")).upper() or None)

        eligible_idx = []
        for i, f in enumerate(flows):
            ft = str(f.get("type", "")).lower()
            fp = str(f.get("product", "")).lower()
            if want_type and not (ft == want_type or fp == want_type):
                continue
            if want_ccy and (str(f.get("currency", "USD")).upper() != want_ccy):
                continue
            eligible_idx.append(i)

        n_take = max(0, int(round(len(eligible_idx) * pct)))
        take_idx = eligible_idx[:n_take]

        touched, mod_ct = [], 0
        for i in take_idx:
            f = flows[i]
            try:
                fd = datetime.fromisoformat(str(f.get("date")).split("T")[0]).date()
            except Exception:
                continue
            sgn = _sign(f)

            if action == "accelerate_outflows" and sgn < 0:
                days = int(effect.get("days", 0))
                mult = float(effect.get("multiplier", 1.0))
                newd = clamp(as_of_d, fd - timedelta(days=abs(days)))
                before_amt = float(f.get("amount", 0.0))
                changed = False
                if newd != fd:
                    f["date"] = newd.isoformat()
                    changed = True
                if mult != 1.0:
                    f["amount"] = before_amt * mult
                    changed = True
                if changed:
                    mod_ct += 1
                    if len(touched) < 5:
                        touched.append({"idx": i, "date": str(fd), "→": str(newd), "amount": before_amt, "→amount": f["amount"]})

            elif action == "delay_inflows" and sgn > 0:
                days = int(effect.get("days", 0))
                newd = clamp(as_of_d, fd + timedelta(days=abs(days)))
                if newd != fd:
                    f["date"] = newd.isoformat()
                    mod_ct += 1
                    if len(touched) < 5:
                        touched.append({"idx": i, "date": str(fd), "→": str(newd), "amount": float(f.get("amount", 0.0))})

            elif action == "haircut_inflows" and sgn > 0:
                mult = float(effect.get("multiplier", 1.0))
                if mult != 1.0:
                    before_amt = float(f.get("amount", 0.0))
                    f["amount"] = before_amt * mult
                    mod_ct += 1
                    if len(touched) < 5:
                        touched.append({"idx": i, "amount": before_amt, "→amount": f["amount"]})

        audits.append({
            "action": action,
            "criteria": {"instrument_type": want_type, "currency": want_ccy},
            "before_eligible": len(eligible_idx),
            "percentage_applied": pct,
            "touched_flows": mod_ct,
            "sample_changes": touched
        })
        audit.append(f"Behavior {action}: eligible={len(eligible_idx)} changed={mod_ct} samples={touched}")

    return flows, audits

# ---------- Adaptive call & rebuild ----------
def _adapt_call(func, **kwargs):
    """
    Adapt function call with parameter aliases.

    Args:
        func: Function to call.
        **kwargs: Arguments.

    Returns:
        Any: Function result.
    """
    sig = inspect.signature(func)
    params = sig.parameters
    ALIASES = {
        "portfolio":        ["portfolio_view", "portfolio_dict"],
        "portfolio_view":   ["portfolio", "portfolio_dict"],
        "exchange_rate":    ["fx_rates", "exchange_rates"],
        "fx_rates":         ["exchange_rate", "exchange_rates"],
        "as_of":            ["asof", "as_of_date"],
        "as_of_date":       ["as_of"],
        "counterparty_data":["counterparty", "counterparties"],
    }
    take = {k: v for k, v in kwargs.items() if k in params}
    for name, p in params.items():
        if name in take:
            continue
        req = (p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)) and (p.default is p.empty)
        if not req:
            continue
        for src in ALIASES.get(name, []):
            if src in kwargs:
                take[name] = kwargs[src]
                break
    return func(**take)

def rebuild_state(as_of_d: date,
                  bank_profile: dict,
                  counterparty: dict | list,
                  fx_rates: dict) -> dict:
    """
    Rebuild state from inputs, injecting flows.

    Args:
        as_of_d: As-of date.
        bank_profile: Bank profile.
        counterparty: Counterparty data.
        fx_rates: FX rates.

    Returns:
        Dict: Rebuilt state.
    """
    from with_whom.portfolio_aggregator import build_portfolio_view
    try:
        portfolio_view = _adapt_call(
            build_portfolio_view,
            bank_profile=bank_profile,
            counterparty_data=counterparty,
            exchange_rate=fx_rates,
            fx_rates=fx_rates,
            as_of=as_of_d.isoformat(),
            as_of_date=as_of_d.isoformat(),
            disable_cache=True, force_rebuild=True, cache=False, cache_bypass=True
        )
    except Exception:
        portfolio_view = _adapt_call(
            build_portfolio_view,
            bank_profile=bank_profile,
            counterparty_data=counterparty,
            exchange_rate=fx_rates,
            fx_rates=fx_rates
        )

    # Extract shocked flows from 'counterparty'
    flows_override: List[dict] = []
    cps = counterparty if isinstance(counterparty, list) else [counterparty]
    for cp in cps:
        if not isinstance(cp, dict):
            continue
        for key in ("cashflows", "cash_flows", "cashflow", "flows", "projected_cashflows"):
            v = cp.get(key)
            if isinstance(v, list):
                flows_override.extend([dict(r) for r in v if isinstance(r, dict)])

    # Force these flows into portfolio_view
    if not isinstance(portfolio_view, dict):
        portfolio_view = {}
    portfolio_view.setdefault("fx_rates", fx_rates or {})
    portfolio_view["as_of"] = as_of_d.isoformat()

    if flows_override:
        portfolio_view["cashflows"] = flows_override
        portfolio_view["counterparties"] = [{"counterparty_id": "ALL", "cashflows": flows_override}]
        preview = portfolio_view.setdefault("cashflows_preview", {})
        preview["scenario"] = flows_override

    try:
        from with_whom.counterpart_aggregator import aggregate_portfolio_intel
        _ = _adapt_call(
            aggregate_portfolio_intel,
            portfolio_view=portfolio_view,
            portfolio=portfolio_view,
            as_of=as_of_d.isoformat(),
            as_of_date=as_of_d.isoformat()
        )
    except Exception:
        pass

    from with_whom.state_builder import build_state
    try:
        state = _adapt_call(
            build_state,
            portfolio=portfolio_view,
            portfolio_view=portfolio_view,
            bank_profile=bank_profile
        )
    except Exception:
        state = build_state(portfolio_view, bank_profile)  # type: ignore

    return state

# ---------- Baseline loader ----------
def _load_baseline_state():
    """
    Load baseline state from state.json or baseline.json.

    Returns:
        Dict: Baseline state.
    """
    s = _read_json(FILES["state_json"], None)
    if isinstance(s, dict) and s:
        return s.get("state", s)
    base_container = _read_json(FILES["baseline_state"], {})
    return base_container.get("state", base_container)

# ---------- NEWS resolver ----------
def resolve_news_refs(just_map: dict, ai_news: dict) -> List[dict]:
    """
    Resolve news references from justification_map.

    Args:
        just_map: Justification map.
        ai_news: AI news dict.

    Returns:
        List: Referenced news items.
    """
    if not ai_news:
        return []
    items = ai_news.get("news", {}).get("news_items") or ai_news.get("news_items") or []
    refs = set()
    for v in (just_map or {}).values():
        for m in re.finditer(r"NEWS\s*\[([0-9,\s]+)\]", str(v)):
            for n in m.group(1).split(","):
                n = n.strip()
                if n.isdigit():
                    idx = int(n)
                    if 0 <= idx < len(items):
                        refs.add(idx)
    return [items[i] for i in sorted(refs)]

# ---------- Adversity guardrail ----------
def enforce_adversity(flows_work: List[dict],
                      stressed_state: dict,
                      baseline_inds: dict,
                      baseline_basis: str,
                      as_of_d: date,
                      audit: List[str],
                      margin: float = ADVERSITY_MARGIN) -> Tuple[List[dict], dict, dict, dict]:
    """
    Enforce adversity by scaling outflows if needed.

    Args:
        flows_work: Cashflows.
        stressed_state: Stressed state.
        baseline_inds: Baseline indicators.
        baseline_basis: Basis for comparison.
        as_of_d: As-of date.
        audit: Audit list.
        margin: Adversity margin (default 0.05).

    Returns:
        Tuple: Adjusted flows, state, comp, guard_meta.
    """
    """
    If comparable LCR >= baseline_LCR, deterministically scale 30-day outflows
    (and lightly haircut inflows) to reach at most baseline_LCR*(1 - margin).
    Rebuild state once and return new flows and recomputed indicators.
    """
    baseline_LCR = baseline_inds.get("lcr_effective")
    if baseline_LCR is None:
        audit.append("[GUARDRAIL] Baseline LCR missing; cannot enforce.")
        return flows_work, stressed_state, {}, {}

    comp0 = compute_lcr_show_basis(stressed_state, force_basis=baseline_basis)
    if comp0["lcr_effective"] is None or comp0["lcr_effective"] < baseline_LCR*(1.0 - margin):
        # Already adverse enough
        return flows_work, stressed_state, comp0, {"applied": False}

    # Target effective LCR
    target = baseline_LCR * (1.0 - margin)
    hqla = comp0["hqlatotal_usd"]
    out_cur = comp0["outflows_30d"]
    in_cur  = comp0["inflows_30d"]

    if baseline_basis == "gross":
        # Need gross_out >= hqla / target
        target_gross = (hqla / target) if target > 0 else out_cur * 1.25
        factor_out = max(1.0, min(2.0, target_gross / max(out_cur, 1e-9)))
        factor_in  = 1.0  # no change needed for gross basis
    else:
        # Need net_out >= hqla / target
        net_cur = max(0.0, out_cur - in_cur)
        target_net = (hqla / target) if target > 0 else net_cur + 1e6
        delta = max(0.0, target_net - net_cur)
        # Prefer increasing outflows
        factor_out = 1.0 + (delta / max(out_cur, 1e-9)) if out_cur > 0 else 1.25
        factor_out = max(1.0, min(2.0, factor_out))
        # Lightly reduce inflows up to 10%
        factor_in = 0.90

    # Apply factors to flows inside 30-day window
    lo, hi = as_of_d + timedelta(days=1), as_of_d + timedelta(days=30)
    touched_out = touched_in = 0
    for f in flows_work:
        try:
            fd = _to_date(f.get("date"))
        except Exception:
            continue
        if not (lo <= fd <= hi):
            continue
        sgn = _sign_from_direction(f)
        if sgn < 0:
            before = float(f.get("amount", 0.0))
            f["amount"] = before * factor_out
            touched_out += 1
        else:
            before = float(f.get("amount", 0.0))
            f["amount"] = before * factor_in
            touched_in += 1

    audit.append(f"[GUARDRAIL] Applied adversity scaling: outflows x{factor_out:.3f}, inflows x{factor_in:.3f} "
                 f"(touched out={touched_out}, in={touched_in}); target LCR <= {target:.6f}")

    return flows_work, None, comp0, {"applied": True, "factor_out": factor_out, "factor_in": factor_in, "target_lcr": target}

# ---------- MAIN ----------
def main():
    """
    Main entry point: Process scenarios and save results.
    """
    # 0) Schema check
    schema_self_check()

    # 1) Load inputs
    raw_scen = _read_json(FILES["standalone_scenarios"], [])
    scenarios = raw_scen.get("scenarios") if isinstance(raw_scen, dict) else raw_scen
    ai_news = _read_json(FILES["ai_news"], {})
    bank_profile = _read_json(FILES["bank_profile"], {})
    counterparty_list = _read_json(FILES["counterparty_data"], [])
    fx_rates = _read_json(FILES["exchange_rate"], {})
    baseline_metrics = _read_json(FILES["baseline_metrics"], {})

    # Flatten all counterparties' cashflows
    all_flows = []
    for cp in counterparty_list:
        cflows = cp.get("cashflows", [])
        if isinstance(cflows, list):
            all_flows.extend(deepcopy(cflows))

    as_of_d = _parse_as_of()
    print(f"[INFO] Using as_of={as_of_d.isoformat()} for all scenarios.")

    # 2) Baseline (validator parity)
    baseline_state = _load_baseline_state()
    baseline_audit = []
    baseline_inds = compute_indicators_from_state(baseline_state, as_of_d, fx_rates, baseline_audit, mode="baseline_show")
    baseline_basis = "gross" if (baseline_inds.get("raw_net_outflows", 1) <= 0) else "net"

    results = {
        "as_of": as_of_d.isoformat(),
        "assumptions": {
            "inflow_cap_stress": INFLOW_CAP_STRESS,
            "basel_caps": {"Level2_max": LEVEL2_MAX_SHARE, "Level2B_max": LEVEL2B_MAX_SHARE},
            "comparison_basis": baseline_basis,
            "guardrail": {"enabled": True, "margin": ADVERSITY_MARGIN}
        },
        "baseline": baseline_inds,
        "scenarios": [],
        "worst_case": None
    }

    # ---------- simple, explicit VaR scaler (PoC) ----------
    def scale_var_simple(baseline_var, rates_bps, credit_bps, fx_shift_pct):
        """
        Transparent heuristic so VaR visibly reacts to shocks.
        - rates_bps: sum(|Δ|)/100 → 'r_mag' in 100bp units
        - credit_bps: sum(|Δ|)/100 → 'c_mag' in 100bp units
        - fx_shift_pct: sum(|Δ|)/10 → 'f_mag' in 10% units
        Coefficients are disclosed in JSON output.
        """
        if baseline_var is None:
            return None
        r_mag = sum(abs(float(x)) for x in (rates_bps or {}).values()) / 100.0
        c_mag = sum(abs(float(x)) for x in (credit_bps or {}).values()) / 100.0
        f_mag = sum(abs(float(x)) for x in (fx_shift_pct or {}).values()) / 10.0
        COEFF = {"rates": 0.25, "credit": 0.15, "fx": 0.10}  # disclose to treasurer
        factor = 1.0 + (COEFF["rates"] * r_mag) + (COEFF["credit"] * c_mag) + (COEFF["fx"] * f_mag)
        return float(baseline_var) * factor, {
            "coefficients": COEFF,
            "magnitudes": {"rates_100bps": r_mag, "credit_100bps": c_mag, "fx_10pct": f_mag},
            "factor": factor
        }

    lcr_vals = []

    # 3) Process scenarios
    for sc in scenarios:
        sid = sc.get("id") or f"scenario_{len(results['scenarios']) + 1:03d}"
        desc = sc.get("description", "")
        shocks = sc.get("shocks", {})
        behavs = sc.get("behaviors", [])

        audit_lines = [f"Scenario {sid}: {desc}"]

        # FX stress
        fx_stressed = apply_fx_shifts(fx_rates, shocks.get("fx_shift_pct"), audit_lines)

        # HQLA haircuts (effective + caps)
        prof_stressed = apply_hqla_haircuts(bank_profile, shocks.get("liquidity_haircut_mult"), audit_lines)

        # Liquidity shocks + behaviors
        flows_work = deepcopy(all_flows)
        flows_work, liq_summary = apply_liquidity_shocks_to_flows(flows_work, shocks, as_of_d, audit_lines)

        # snapshot BEFORE behaviors (used for per-behavior attribution)
        flows_after_shocks_only = deepcopy(flows_work)

        flows_work, beh_summary = apply_behaviors(flows_work, behavs, as_of_d, audit_lines)

        # ---------- NEW: enrich behaviors_applied with numeric USD metrics and a ready treasurer sentence ----------
        try:
            # tiny local helpers (no API changes)
            def _rate(ccy: str) -> float:
                c = (ccy or "USD").upper()
                try:
                    return float(fx_stressed.get(c, 1.0)) if c != "USD" else 1.0
                except Exception:
                    return 1.0

            def _abs_usd(flow: dict) -> float:
                try:
                    return abs(float(flow.get("amount", 0.0))) * _rate(flow.get("currency", "USD"))
                except Exception:
                    return 0.0

            def _in_window(d) -> bool:
                try:
                    dt = datetime.fromisoformat(str(d).split("T")[0]).date()
                except Exception:
                    return False
                return (as_of_d < dt <= (as_of_d + timedelta(days=30)))

            def _sgn(flow: dict) -> int:
                return _sign_from_direction(flow)

            def _type_match(flow: dict, want_type: str) -> bool:
                return _flow_type_matches(flow, want_type) if want_type else True

            def _currency_match(flow: dict, want_ccy: Optional[str]) -> bool:
                if not want_ccy:
                    return True
                return str(flow.get("currency", "USD")).upper() == str(want_ccy).upper()

            def _pretty_instr(s: str) -> str:
                s = (s or "").strip().replace("_", " ").title()
                return s if s.endswith("s") or s == "" else (s + "s")

            def _fmt_money_short(x: float) -> str:
                ax = abs(float(x))
                if ax >= 1e9:
                    return f"${ax / 1e9:,.3f}bn"
                if ax >= 1e6:
                    return f"${ax / 1e6:,.0f}m"
                return f"${ax:,.0f}"

            def _fmt_bps(x: Optional[float]) -> str:
                if x is None:
                    return "n/a"
                return f"{float(x):+,.1f} bps"

            # map once for faster lookups
            pre_len = len(flows_after_shocks_only)
            post_len = len(flows_work)
            same_length = (pre_len == post_len)

            # per-behavior computation uses a fresh single-behavior application to isolate impact
            enriched = []
            for idx, b in enumerate(behavs or []):
                # base info
                action = (b.get("action") or "").strip().lower()
                crit = b.get("criteria") or {}
                scope = b.get("scope") or {}
                eff = b.get("effect") or {}

                want_type = str(crit.get("instrument_type") or crit.get("product") or "").strip().lower()
                want_ccy = (crit.get("currency") or None)
                # percentage normalization like engine
                pct = float(scope.get("percentage", 1.0))
                if pct > 1.0: pct /= 100.0
                pct = max(0.0, min(1.0, pct))
                scope_pct = pct * 100.0

                # sign requirement per action (mirror apply_behaviors semantics)
                if action == "accelerate_outflows":
                    need_sign = -1
                elif action in ("delay_inflows", "haircut_inflows"):
                    need_sign = +1
                else:
                    need_sign = 0  # unknown → treat as both

                # eligible set (from flows AFTER shocks, BEFORE behaviors), no date filter
                eligible_idx = []
                eligible_usd_sum = 0.0
                for i, f in enumerate(flows_after_shocks_only):
                    if want_type and not _type_match(f, want_type):
                        continue
                    if not _currency_match(f, want_ccy):
                        continue
                    if need_sign and (_sgn(f) != need_sign):
                        continue
                    eligible_idx.append(i)
                    eligible_usd_sum += _abs_usd(f)

                # touched set by reapplying ONLY this behavior on a copy of the pre-behavior flows
                pre_copy = deepcopy(flows_after_shocks_only)
                tmp_audit = []
                post_single, _one = apply_behaviors(pre_copy, [b], as_of_d, tmp_audit)

                touched_count = 0
                touched_usd_sum = 0.0
                if same_length:
                    for i, (f0, f1) in enumerate(zip(flows_after_shocks_only, post_single)):
                        # only consider flows in eligible set for this behavior
                        if i not in eligible_idx:
                            continue
                        changed = False
                        # date or amount change triggers "touched"
                        try:
                            d0 = str(f0.get("date"))
                            d1 = str(f1.get("date"))
                        except Exception:
                            d0 = d1 = ""
                        try:
                            a0 = float(f0.get("amount", 0.0))
                            a1 = float(f1.get("amount", 0.0))
                        except Exception:
                            a0 = a1 = 0.0
                        if (d0 != d1) or (abs(a0 - a1) > 1e-12):
                            changed = True
                        if changed:
                            touched_count += 1
                            touched_usd_sum += _abs_usd(f1)

                # net USD moved into the 30-day window (eligible flows only)
                pre_win_sum = 0.0
                post_win_sum = 0.0
                for i, (f0, f1) in enumerate(zip(flows_after_shocks_only, post_single) if same_length else []):
                    if i not in eligible_idx:
                        continue
                    if _in_window(f0.get("date")):
                        pre_win_sum += _abs_usd(f0)
                    if _in_window(f1.get("date")):
                        post_win_sum += _abs_usd(f1)
                net_into_30d_usd = post_win_sum - pre_win_sum

                # behavior-only comparable LCR delta (basis = baseline_basis), with scenario shocks in place
                # build a state using ONLY this behavior (no cross-behavior contamination)
                counterparty_one = [{"counterparty_id": "ALL", "cashflows": post_single}]
                state_one = rebuild_state(as_of_d, prof_stressed, counterparty_one, fx_stressed)
                comp_one = compute_lcr_show_basis(state_one, force_basis=baseline_basis)
                base_eff = baseline_inds.get("lcr_effective")
                if comp_one.get("lcr_effective") is not None and base_eff is not None:
                    lcr_delta_bps = (float(comp_one["lcr_effective"]) - float(base_eff)) * 10000.0
                else:
                    lcr_delta_bps = None

                # pretty labels
                instr_label = _pretty_instr(want_type) if want_type else "All Instruments"
                ccy_label = (str(want_ccy).upper() if want_ccy else "ALL")
                days = int(eff.get("days", 0) or 0)
                mult = float(eff.get("multiplier", 1.0) or 1.0)
                # unicode minus and times to match your examples
                sign_days = "−" if days < 0 else ("+" if days > 0 else "±")
                effect_pretty = f"{sign_days}{abs(days)} d, ×{mult:.2f}"

                # treasurer sentence
                action_pretty = {
                    "accelerate_outflows": "Accelerate Outflows",
                    "delay_inflows": "Delay Inflows",
                    "haircut_inflows": "Haircut Inflows",
                }.get(action, (action or "Behavior").replace("_", " ").title())

                sent = (
                    f"{action_pretty} — {ccy_label} — {instr_label}: "
                    f"Scope {scope_pct:.0f}% of eligible ({len(eligible_idx)} flows; {_fmt_money_short(eligible_usd_sum)} eligible); "
                    f"{touched_count} flows changed ({_fmt_money_short(touched_usd_sum)} touched); "
                    f"effect {effect_pretty}; "
                    f"moved {_fmt_money_short(net_into_30d_usd)} into the 30-day window; "
                    f"LCR Δ {_fmt_bps(lcr_delta_bps)} (basis: {baseline_basis})."
                )

                # write back into behaviors_applied entry (keep existing keys, only enrich)
                if idx < len(beh_summary):
                    beh_summary[idx]["before_eligible_usd"] = eligible_usd_sum
                    beh_summary[idx]["touched_usd"] = touched_usd_sum
                    beh_summary[idx]["net_into_30d_usd"] = net_into_30d_usd
                    beh_summary[idx]["lcr_delta_bps"] = lcr_delta_bps
                    beh_summary[idx]["instrument_label"] = instr_label
                    beh_summary[idx]["currency"] = ccy_label
                    beh_summary[idx]["scope_pct"] = scope_pct
                    beh_summary[idx]["effect_pretty"] = effect_pretty
                    beh_summary[idx]["treasurer_sentence"] = sent

                enriched.append(sent)

            if enriched:
                audit_lines.append("[BEHAVIOR-SUMMARY] " + " | ".join(enriched))
        except Exception as _e:
            audit_lines.append(f"[WARN] Behavior enrichment failed: {_e}")

        # Rebuild state once (with all behaviors applied)
        stressed_state = rebuild_state(as_of_d, prof_stressed, [{"counterparty_id": "ALL", "cashflows": flows_work}],
                                       fx_stressed)

        # Indicators (stressed PoC: 75% cap)
        stressed_inds = compute_indicators_from_state(stressed_state, as_of_d, fx_stressed, audit_lines,
                                                      mode="stressed")

        # Comparable LCR on SAME basis as baseline (fair comparison)
        comp = compute_lcr_show_basis(stressed_state, force_basis=baseline_basis)
        stressed_inds["lcr_effective_comparable"] = comp["lcr_effective"]
        stressed_inds["lcr_effective_gross"] = comp["lcr_gross"]
        stressed_inds["lcr_effective_cap_net"] = stressed_inds.get("lcr_effective")
        stressed_inds["comparison_basis"] = baseline_basis

        # ---- VaR scaling (simple, disclosed) ----
        base_var = baseline_inds.get("var_1d_99_usd")
        var_scaled, var_meta = scale_var_simple(
            base_var,
            shocks.get("rates_bps", {}),
            shocks.get("credit_spread_bps", {}),
            shocks.get("fx_shift_pct", {})
        )
        if var_scaled is not None:
            stressed_inds["var_1d_99_usd"] = var_scaled

        # ---- Adversity guardrail (keeps comparable LCR ≤ baseline*(1-5%)) ----
        guardrail_info = {"applied": False}
        if (baseline_inds.get("lcr_effective") is not None and
            (stressed_inds["lcr_effective_comparable"] is None or
             stressed_inds["lcr_effective_comparable"] >= baseline_inds["lcr_effective"] * (1.0 - ADVERSITY_MARGIN))):
            flows_guard, _, comp0, guard_meta = enforce_adversity(
                flows_work, stressed_state, baseline_inds, baseline_basis, as_of_d, audit_lines, ADVERSITY_MARGIN
            )
            guardrail_info = guard_meta or {"applied": False}
            if guardrail_info.get("applied"):
                # Rebuild after guardrail
                counterparty_guard = [{"counterparty_id": "ALL", "cashflows": flows_guard}]
                stressed_state = rebuild_state(as_of_d, prof_stressed, counterparty_guard, fx_stressed)
                stressed_inds  = compute_indicators_from_state(stressed_state, as_of_d, fx_stressed, audit_lines, mode="stressed")
                comp           = compute_lcr_show_basis(stressed_state, force_basis=baseline_basis)
                stressed_inds["lcr_effective_comparable"] = comp["lcr_effective"]
                stressed_inds["lcr_effective_gross"]      = comp["lcr_gross"]
                stressed_inds["lcr_effective_cap_net"]    = stressed_inds.get("lcr_effective")
                stressed_inds["comparison_basis"]         = baseline_basis
                # re-apply VaR scaling (it depends on shock magnitudes, not flows)
                var_scaled, var_meta = scale_var_simple(
                    base_var,
                    shocks.get("rates_bps", {}),
                    shocks.get("credit_spread_bps", {}),
                    shocks.get("fx_shift_pct", {})
                )
                if var_scaled is not None:
                    stressed_inds["var_1d_99_usd"] = var_scaled

        # Diffs & sanity flags
        diffs = {k: (stressed_inds.get(k) - baseline_inds.get(k) if (stressed_inds.get(k) is not None and baseline_inds.get("k") is not None) else None)
                 for k in ["hqlatotal_usd","gross_outflows_30d_usd","inflows_30d_usd","net_outflows_30d_usd","lcr_net","lcr_gross","lcr_effective","var_1d_99_usd"]}
        if stressed_inds.get("lcr_effective_comparable") is not None and baseline_inds.get("lcr_effective") is not None:
            diffs["lcr_effective_comparable"] = stressed_inds["lcr_effective_comparable"] - baseline_inds["lcr_effective"]

        sanity_flags = {}
        if (stressed_inds.get("lcr_effective_comparable") is not None and baseline_inds.get("lcr_effective") is not None):
            if stressed_inds["lcr_effective_comparable"] >= baseline_inds["lcr_effective"]:
                sanity_flags["comparable_lcr_not_worse"] = True
                audit_lines.append(f"[SANITY] Comparable LCR did not worsen: baseline {baseline_inds['lcr_effective']:.3f} vs stressed {stressed_inds['lcr_effective_comparable']:.3f}")

        # Transparency fields
        explanation = sc.get("explanation")
        just_map = sc.get("justification_map") or {}
        news_refs = resolve_news_refs(just_map, ai_news)
        inputs_used = {
            "hqla_effective": prof_stressed.get("liquidity", {}).get("hqla_effective"),
            "per_currency_LCR": baseline_metrics.get("per_currency_LCR"),
        }

        # Sidecar audit
        sidecar = LOG_DIR / f"scenario_{sid}_audit.json"
        _write_json(sidecar, {"scenario_id": sid, "audit": audit_lines})

        sc_result = {
            "id": sid,
            "description": desc,
            "explanation": explanation,
            "justification_map": just_map,
            "news_refs": news_refs,
            "inputs_used": inputs_used,
            "assumptions": {"var_scaling": var_meta, "guardrail": guardrail_info},
            "shocks_applied": {
                "fx_shift_pct": shocks.get("fx_shift_pct"),
                "rates_bps": shocks.get("rates_bps"),
                "credit_spread_bps": shocks.get("credit_spread_bps"),
                "liquidity_haircut_mult": { _norm_level(k): v for k, v in (shocks.get("liquidity_haircut_mult") or {}).items() },
                "liquidity_shocks_summary": liq_summary,
            },
            "behaviors_applied": beh_summary,
            "stressed_indicators": stressed_inds,
            "differences": diffs,
            "sanity_flags": sanity_flags or {},
            "audit_log_sidecar": str(sidecar),
        }
        results["scenarios"].append(sc_result)

        lcr = stressed_inds.get("lcr_effective_comparable")
        if lcr is not None:
            lcr_vals.append((sid, float(lcr)))

    # 4) Worst-case summary
    worst = None
    if lcr_vals:
        sid_min, lcr_min = min(lcr_vals, key=lambda x: x[1])
        baseline_comp = baseline_inds.get("lcr_effective")
        worse_than_baseline = (baseline_comp is not None and lcr_min < baseline_comp)
        worst = {
            "scenario_id": sid_min,
            "lcr_worst": lcr_min,
            "worse_than_baseline": bool(worse_than_baseline),
            "comparison_basis": baseline_basis
        }
    results["worst_case"] = worst

    # 5) Save
    _write_json(OUT_MAIN, results)
    print(f"[OK] Wrote {OUT_MAIN} and per-scenario audits in {LOG_DIR}")


if __name__ == "__main__":
    main()