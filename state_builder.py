# state_builder.py
"""
State builder for PoC (updated to carry through market_risk exposures/vols/corr).
Public signature preserved: build_state(portfolio: Dict, bank_profile: Dict, shocks: Dict=None)
CLI writes: with_whom/data/state.json

Module Overview:
----------------
This module constructs the canonical "state" object central to the Treasury AI Proof-of-Concept (PoC).
It normalizes inputs from portfolio aggregation and bank profile, computes Basel III-aligned liquidity metrics
(LCR, survival days), and integrates market risk parameters for downstream stress testing and hedging.

Key Features:
-------------
- Robust normalization of HQLA with effective values post-haircut for stressed scenarios.
- Harvesting of cashflows from heterogeneous portfolio structures (counterparty-level, previews, simulations).
- Computation of inflows/outflows over 30-day LCR horizon, with PoC-specific inflow discounting under stress.
- Passthrough of risk exposures, volatilities, correlations, and VaR for ALM/risk integration.
- Audit trails for sanitization (e.g., negative HQLA clipped to zero) and assumptions (e.g., net vs. gross LCR).

Regulatory Alignment:
---------------------
- LCR per Basel III (BCBS 238): HQLA / net outflows >= 100%; uses 20% inflow haircut for stress conservatism.
- Supports NSFR-like survival days estimation (HQLA / daily net outflow).
- Risk params enable VaR computation under ECB/SEC stress guidelines (e.g., +200bps rate shock).

Dependencies:
-------------
- Python 3.8+: typing, datetime, collections, json, pathlib (standard library).
- pandas: DataFrame handling for cashflow normalization.
- Optional: cf_normalizer (project-specific; fallback to basic pandas coercion).

Usage in Workflow:
------------------
- Upstream: portfolio_aggregator.py → portfolio_view.json.
- This: build_state() → state.json.
- Downstream: stress_tester.py (applies shocks to state), hedge_proposer.py (uses exposures/limits).
- CLI: python state_builder.py (loads inputs, writes state.json).

Error Handling:
---------------
- Graceful fallbacks for missing data (e.g., default USD rates, zero reserves).
- Exceptions in normalization logged but do not halt; empty DataFrames preserved.
- Audit dict captures anomalies (e.g., negative net outflows) for treasurer review.

Author:
-------
FintekMinds TreasuryAI PoC — Liquidity & Risk Normalization Module
Version: 2025-10 (Updated for market_risk integration)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path

import pandas as pd

# Try to import a project normalize_cashflows; if not available provide a safe fallback.
# This ensures PoC runs in isolation; full normalizer handles FX, direction signing, bucketing.
try:
    from cf_normalizer import normalize_cashflows  # type: ignore
except Exception:
    try:
        from with_whom.cf_normalizer import normalize_cashflows  # type: ignore
    except Exception:
        def normalize_cashflows(raw_cfs, portfolio, bank_profile, as_of_date):
            """
            Fallback cashflow normalizer (Pandas-only).

            Args:
                raw_cfs: List of raw cashflow dicts from harvested sources.
                portfolio: Original portfolio dict (unused in fallback).
                bank_profile: Normalized bank profile (unused in fallback).
                as_of_date: Reference datetime for horizon (unused in fallback).

            Returns:
                pd.DataFrame: Basic DataFrame with 'date' coerced to datetime if present.

            Notes:
                Minimal for PoC; lacks regulatory bucketing (e.g., 30d LCR), FX conversion.
                In production, use full cf_normalizer for compliance (e.g., signed amounts: +inflow/-outflow).
            """
            df = pd.DataFrame(raw_cfs)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            return df


# DATA_DIR points to with_whom/data (sibling of this module)
# Ensures consistent I/O for CLI and module imports; creates if absent.
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _to_date(x) -> datetime:
    """
    Coerce input to Python datetime.

    Args:
        x: str, pd.Timestamp, or datetime-like.

    Returns:
        datetime: Standardized datetime object.

    Purpose:
        Unifies date parsing from JSON/pandas for accurate horizon bucketing in LCR calcs.
        Handles ISO strings and timestamps common in portfolio exports.
    """
    if isinstance(x, str):
        return pd.to_datetime(x).to_pydatetime()
    if isinstance(x, pd.Timestamp):
        return x.to_pydatetime()
    return x


def normalize_bank_profile(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize bank_profile for build_state.

    Back-compatible: keep per-level 'amount_usd' unchanged (raw), but if
    bank_profile.liquidity.hqla_effective is present, also attach
    'amount_usd_effective' per level so downstream can sum stressed effective
    amounts without breaking other code.

    Args:
        raw: Raw dict from bank_profile.json (may have legacy keys).

    Returns:
        Dict[str, Any]: Normalized profile with floats, standard keys, computed fields.

    Process:
        - Intraday liquidity: Fallback to reserve/currency if missing.
        - HQLA: Normalize levels (Level 1/2A/2B), attach effective post-haircut if provided.
        - Risk: Passthrough exposures/vols/corrs/VaR from market_risk block.
        - Survival days: Compute from HQLA/net_outflows if absent (30-day LCR horizon).
        - Metadata: Track sources/overrides for audit.

    Compliance:
        - HQLA levels/haircuts per Basel III Annex 1 (0%/15%/50%).
        - Ensures positive values; negatives sanitized downstream.
    """
    bp = raw or {}
    normalized: Dict[str, Any] = {}

    # Intraday (unchanged)
    # Provides immediate liquidity buffer for T+0 gaps.
    intraday = bp.get("intraday_liquidity")
    if not intraday:
        liquidity_block = bp.get("liquidity", {})
        intraday = {
            "reserve": liquidity_block.get("reserve", 0.0),
            "currency": liquidity_block.get("currency", liquidity_block.get("reporting_currency", "USD"))
        }
    normalized["intraday_liquidity"] = {
        "reserve": float(intraday.get("reserve", 0.0) or 0.0),
        "currency": intraday.get("currency", "USD")
    }

    # HQLA: build from breakdown; add amount_usd_effective if hqla_effective provided
    # Supports stressed HQLA (e.g., post-market shock haircuts) for scenario analysis.
    def _norm_level(lvl: str) -> str:
        """
        Standardize HQLA level strings to Basel III format.

        Args:
            lvl: Raw string (e.g., 'l1', 'Level2B').

        Returns:
            str: Normalized (e.g., 'Level 1').
        """
        s = str(lvl).strip().lower().replace(" ", "")
        if s in {"level1", "l1"}: return "Level 1"
        if s in {"level2a", "l2a"}: return "Level 2A"
        if s in {"level2b", "l2b"}: return "Level 2B"
        return str(lvl).replace("level", "Level")

    liquidity_block = bp.get("liquidity", {}) or {}
    breakdown = liquidity_block.get("hqla_breakdown") or {}
    effective_map = liquidity_block.get("hqla_effective") or {}
    eff_by_norm = {}
    if isinstance(effective_map, dict):
        for k, v in effective_map.items():
            try:
                eff_by_norm[_norm_level(k)] = float(v)
            except Exception:
                pass

    hqla_list: List[Dict[str, Any]] = []
    if isinstance(breakdown, dict) and breakdown:
        for lvl_raw, node in breakdown.items():
            lvl = _norm_level(lvl_raw)
            info = node or {}
            # keep raw amount_usd as-is (back-compat)
            try:
                amt_raw = float(info.get("amount_usd", info.get("amount", 0.0)) or 0.0)
            except Exception:
                amt_raw = 0.0
            haircut = info.get("haircut", None)
            ccy = info.get("currency") or liquidity_block.get("currency") or "USD"
            item = {"level": lvl, "amount_usd": amt_raw, "haircut": haircut, "currency": ccy}
            # add effective amount if present from stress
            if lvl in eff_by_norm:
                item["amount_usd_effective"] = eff_by_norm[lvl]
            hqla_list.append(item)

    if not hqla_list and liquidity_block.get("total_hqla_usd") is not None:
        # fallback aggregate
        try:
            amt = float(liquidity_block.get("total_hqla_usd") or 0.0)
        except Exception:
            amt = 0.0
        hqla_list.append({"level": "aggregate", "amount_usd": amt, "haircut": None, "currency": "USD"})

    normalized["hqla"] = hqla_list

    # Other blocks (unchanged)
    # Limits for counterparty exposure checks in hedging.
    normalized["limits"] = bp.get("limits", bp.get("counterparty_limits", {}))
    # Exposures for VaR/EoE calcs (e.g., FX delta, IR PV01).
    normalized["risk_exposures"] = bp.get("risk_exposures", bp.get("market_risk", {}).get("exposures", {}))
    normalized["risk_vols"] = bp.get("risk_vols", bp.get("market_risk", {}).get("vols", {}))
    normalized["risk_corr"] = bp.get("risk_corr", bp.get("market_risk", {}).get("corr", None))
    normalized["profile_var"] = bp.get("market_risk", {}).get("var", {})

    max_days = bp.get("max_survival_days")
    if max_days is None:
        liquidity_block = bp.get("liquidity", {})
        override = liquidity_block.get("survival_days_override") or bp.get("survival_days_override")
        if override:
            max_days = override
        else:
            try:
                total_hqla = float(liquidity_block.get("total_hqla_usd") or 0.0)
                net_out = float(liquidity_block.get("net_outflows_30d_usd") or 0.0)
                max_days = 30.0 * total_hqla / net_out if net_out > 0 else bp.get("max_survival_days", 120)
            except Exception:
                max_days = bp.get("max_survival_days", 120)
    normalized["max_survival_days"] = float(max_days or 120)

    normalized["simulated_baseline"] = bp.get("simulated_baseline", {})
    governance = bp.get("governance", {}) or {}
    normalized["poc_inflow_fraction"] = float((governance.get("poc") or {}).get("inflow_fraction_available_under_stress", 0.20))
    normalized["_raw_profile_meta"] = {"meta": bp.get("meta", {}), "original_keys": list(bp.keys())}
    # Optional hint for debugging
    normalized["_hqla_source"] = "hqla_effective_present" if eff_by_norm else "hqla_breakdown_only"
    return normalized




def _harvest_cashflows(portfolio: Dict[str, Any], bank_profile: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Harvest cashflows from varied portfolio structures.

    Args:
        portfolio: Dict from portfolio_view.json (nested keys possible).
        bank_profile: Optional; appends simulated cashflows if present.

    Returns:
        List[Dict[str, Any]]: Flattened cashflow records, with counterparty_id propagated.

    Process:
        - Scans top-level/nested keys ('cashflows', 'projected_cashflows', etc.).
        - Handles counterparty arrays, adding IDs to flows.
        - Includes preview stages (baseline/shocked) for scenario chaining.
        - Augments with simulated baselines for PoC completeness.

    Notes:
        Robust to schema drift (e.g., Murex vs. Calypso exports).
        Post-harvest: normalized via cf_normalizer for LCR bucketing.
        Assumes amounts in local currency; FX applied in build_state.
    """
    rows: List[Dict[str, Any]] = []
    # multiple locations for cashflows
    for key in ("cashflows", "cash_flows", "cashflow", "flows", "projected_cashflows"):
        v = portfolio.get(key)
        if isinstance(v, list):
            rows.extend([dict(r) for r in v if isinstance(r, dict)])
    for parent in ("view", "data", "portfolio", "report"):
        d = portfolio.get(parent)
        if isinstance(d, dict):
            for key in ("cashflows", "cash_flows", "cashflow", "flows", "projected_cashflows"):
                v = d.get(key)
                if isinstance(v, list):
                    rows.extend([dict(r) for r in v if isinstance(r, dict)])
    for cp_key in ("counterparties", "cp_data", "counterparty_data"):
        cps = portfolio.get(cp_key)
        if isinstance(cps, list):
            for cp in cps:
                cp_id = cp.get("counterparty_id") or cp.get("name") or ""
                for key in ("cashflows", "cash_flows", "cashflow", "flows"):
                    v = cp.get(key)
                    if isinstance(v, list):
                        for r in v:
                            if isinstance(r, dict):
                                rr = dict(r)
                                rr.setdefault("counterparty_id", cp_id)
                                rows.append(rr)
    pv = portfolio.get("cashflows_preview")
    if isinstance(pv, dict):
        for stage in ("baseline", "scenario", "shocked", "after", "final", "chosen", "candidate"):
            v = pv.get(stage)
            if isinstance(v, list):
                rows.extend([dict(r) for r in v if isinstance(r, dict)])
        for stage, obj in pv.items():
            if isinstance(obj, dict) and isinstance(obj.get("cashflows"), list):
                rows.extend([dict(r) for r in obj["cashflows"] if isinstance(r, dict)])
    # include simulated baseline from bank_profile if present
    if bank_profile and "simulated_baseline" in bank_profile:
        sim_cf = bank_profile["simulated_baseline"].get("cashflows", [])
        if isinstance(sim_cf, list):
            rows.extend([dict(cf) for cf in sim_cf if isinstance(cf, dict)])
    return rows


def build_state(portfolio: Dict[str, Any], bank_profile: Dict[str, Any], shocks: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Build the canonical state used by the rest of the PoC.

    Key additions:
      - Populates state['risk_exposures'], state['risk_vols'], state['risk_corr']
        from bank_profile.market_risk if present.
      - Copies bank_profile.market_risk.var to state['profile_var'] for traceability.

    Args:
        portfolio: Aggregated portfolio_view.json (positions, cashflows, compliance).
        bank_profile: Raw bank_profile.json (liquidity, risks, limits).
        shocks: Optional stress dict (unused in baseline; for future extension).

    Returns:
        Dict[str, Any]: Canonical state with normalized/computed fields.
                       Structure: as_of, hqla, cashflows, liquidity metrics, risks, _audit.

    Process:
        1. Normalize profile (HQLA, risks).
        2. Harvest/normalize cashflows to list-of-dicts.
        3. Sanitize HQLA (clip negatives, audit).
        4. Derive positions/notionals via FX if missing.
        5. Compute 30d inflows/outflows/LCR (net/gross variants).
        6. Apply PoC inflow discount (20%) for stressed assumption.
        7. Integrate risks for ALM.

    Output Fields:
        - Liquidity: inflows_30d_usd, net_outflows_30d_usd, lcr_computed_net, survival_days_assumed.
        - Risks: risk_exposures (e.g., {'FX': {'EUR': {'pos_usd': 1e6}}}), risk_vols, etc.
        - Audit: _audit (negatives, assumptions), _bank_profile_meta.

    Compliance/Assumptions:
        - LCR: Basel III net (75% runoff on unsecured); PoC uses 20% inflow availability.
        - Survival days: HQLA / (net_outflows/30) for NSFR horizon.
        - FX: From portfolio.fx_rates or 1.0 default.
        - Shocks: Placeholder; apply in stress_tester.py (e.g., vol upshock → HQLA effective down).

    Integration:
        - Feeds Streamlit dashboard for before/after LCR visuals.
        - Enables NLP queries (e.g., "Hedge EUR exposure under +100bps shock").
    """
    state: Dict[str, Any] = {}
    state["as_of"] = portfolio.get("as_of") or datetime.today().isoformat()[:10]

    # normalize profile
    bank_profile_norm = normalize_bank_profile(bank_profile or {})

    # collect cashflows and normalize
    raw_cfs = _harvest_cashflows(portfolio, bank_profile_norm)
    as_of_date = _to_date(state["as_of"])
    try:
        df = normalize_cashflows(raw_cfs, portfolio, bank_profile_norm, as_of_date)
    except Exception as e:
        print(f"[STATE_BUILDER] normalize_cashflows raised {type(e).__name__}: {e}")
        df = pd.DataFrame()
    if isinstance(df, pd.DataFrame) and not df.empty:
        normalized_cfs: List[Dict[str, Any]] = df.to_dict(orient="records")
    else:
        normalized_cfs = [dict(r) for r in raw_cfs] if raw_cfs else []

    state["cashflows"] = normalized_cfs

    # sanitize HQLA (zero negative amounts, keep audit)
    # Ensures regulatory positivity; audits for data quality issues.
    hqla_list = bank_profile_norm.get("hqla", []) or []
    sanitized_hqla = []
    negatives = []
    for h in hqla_list:
        try:
            amt = float(h.get("amount_usd") or h.get("amount") or 0.0)
        except Exception:
            amt = 0.0
        if amt < 0:
            negatives.append({"entry": h, "original_amount": amt})
            h["amount_usd"] = 0.0
            amt = 0.0
        h.setdefault("currency", h.get("ccy", "USD"))
        h["amount_usd"] = float(amt)
        sanitized_hqla.append(h)
    state["hqla"] = sanitized_hqla
    if negatives:
        state["_audit"] = state.get("_audit", {})
        state["_audit"]["hqla_negatives_sanitized"] = negatives

    # copy other profile pieces
    state["limits"] = bank_profile_norm.get("limits", {})
    state["intraday_liquidity"] = bank_profile_norm.get("intraday_liquidity", {})

    # basic portfolio fields
    state["positions_gross_usd"] = portfolio.get("positions_gross_usd", 0.0)
    state["total_notional_usd"] = portfolio.get("total_notional_usd", 0.0)
    state["cp_compliance"] = portfolio.get("cp_compliance", [])
    state["fx_rates"] = portfolio.get("fx_rates", {})

    # derive positions if missing
    # Fallback summation of exposures via FX for completeness.
    if not state["positions_gross_usd"] or not state["total_notional_usd"]:
        fx_rates = state.get("fx_rates", {}) or {}
        def _usd(n, ccy):
            try:
                r = float(fx_rates.get(str(ccy).upper(), 1.0))
                return float(n or 0.0) * r
            except Exception:
                return 0.0
        exp_total = 0.0
        for e in (portfolio.get("exposures") or []):
            exp_total += _usd(e.get("notional"), e.get("currency"))
        state["positions_gross_usd"] = abs(exp_total)
        state["total_notional_usd"] = abs(exp_total)

    # -------------------- NEW: bring through market risk inputs --------------------
    # risk_exposures expected shape: {"FX": {"EUR": {"pos_usd":..}}, "IR": {"USD": {"pv01_usd":..}}, ...}
    # Enables downstream VaR: exposure * vol * sqrt(t) * corr matrix.
    state["risk_exposures"] = bank_profile_norm.get("risk_exposures", {}) or {}
    state["risk_vols"] = bank_profile_norm.get("risk_vols", {}) or {}
    state["risk_corr"] = bank_profile_norm.get("risk_corr", None)
    state["profile_var"] = bank_profile_norm.get("profile_var", {}) or {}

    # Keep a copy of the raw profile meta for traceability
    state["_bank_profile_meta"] = bank_profile_norm.get("_raw_profile_meta", {})

    state["max_survival_days"] = bank_profile_norm.get("max_survival_days", 120)
    state["risk_exposures"] = state.get("risk_exposures", {}) or {}
    state["risk_vols"] = state.get("risk_vols", {}) or {}
    state["risk_corr"] = state.get("risk_corr", None)

    # ------------------ PoC enrichment: compute inflows/outflows and LCRs ------------------
    # Buckets cashflows to 30d horizon; applies FX for USD consistency.
    as_of = state.get("as_of") or datetime.today().isoformat()[:10]
    as_of_dt = datetime.fromisoformat(str(as_of)[:10])
    horizon_end = as_of_dt + timedelta(days=30)

    inflows = outflows = 0.0
    fx = state.get("fx_rates", {}) or {}

    for cf in state.get("cashflows", []):
        try:
            dt = pd.to_datetime(cf.get("date") or cf.get("value_date") or as_of_dt).to_pydatetime()
        except Exception:
            dt = as_of_dt
        if not (as_of_dt <= dt <= horizon_end):
            continue
        amt = float(cf.get("amount") or 0.0)
        ccy = str(cf.get("currency") or cf.get("ccy") or "USD").upper()
        try:
            rate = float(fx.get(ccy) or fx.get(ccy.upper()) or 1.0)
        except Exception:
            rate = 1.0
        usd = amt * rate
        direction = str(cf.get("direction") or "").lower()
        if direction in ("out", "pay", "debit", "payout", "-1", "-") or usd < 0:
            outflows += abs(usd)
        else:
            inflows += abs(usd)

    gross_outflows = outflows
    raw_net = outflows - inflows  # raw signed net (can be negative)
    # Clip net_outflows to be non-negative for LCR math; record raw negative value for audit
    net_outflows = max(0.0, raw_net)
    if raw_net < 0:
        state["_audit"] = state.get("_audit", {})
        state["_audit"]["raw_net_outflows_negative"] = {"raw_net_outflows": raw_net, "note": "inflows exceeded outflows in 30d window"}

    # HQLA total
    # Sums raw amounts; effective used in stressed recompute.
    hqla_total = 0.0
    for h in state.get("hqla", []):
        try:
            hqla_total += float(h.get("amount_usd") or 0.0)
        except Exception:
            continue

    state["inflows_30d_usd"] = inflows
    state["outflows_30d_usd"] = outflows
    state["gross_outflows_30d_usd"] = gross_outflows
    state["net_outflows_30d_usd"] = net_outflows
    state["raw_net_outflows_signed"] = raw_net
    state["total_hqla_usd"] = hqla_total

    # computed LCRs: net only when net_outflows > 0; gross always when gross_outflows > 0
    # Provides dual views: conservative gross for high-inflow scenarios.
    state["lcr_computed_net"] = (hqla_total / net_outflows) if net_outflows > 0 else None
    state["survival_days_computed_net"] = (30.0 * hqla_total / net_outflows) if net_outflows > 0 else None
    state["lcr_computed_gross"] = (hqla_total / gross_outflows) if gross_outflows > 0 else None
    state["survival_days_computed_gross"] = (30.0 * hqla_total / gross_outflows) if gross_outflows > 0 else None

    # PoC assumption: inflow_fraction to discount inflows under stress
    # 20% availability mimics behavioral runoff (e.g., 80% counterparty drawdown).
    f = float(bank_profile_norm.get("poc_inflow_fraction", 0.20))
    net_out_assumed = max(0.0, gross_outflows - inflows * f)
    poc = {
        "inflow_fraction_used": f,
        "net_outflows_assumed_usd_raw": gross_outflows - inflows * f,
    }

    # If net_out_assumed is <= 0 => conservative fallback to gross-outflows LCR (no netting)
    if net_out_assumed <= 0:
        poc["net_outflows_assumed_usd"] = None
        poc["lcr_assumed"] = state["lcr_computed_gross"]
        poc["survival_days_assumed"] = state["survival_days_computed_gross"]
        poc["note"] = "inflows too large after discounting; using conservative gross-outflow LCR (no netting) as fallback"
    else:
        poc["net_outflows_assumed_usd"] = net_out_assumed
        poc["lcr_assumed"] = (hqla_total / net_out_assumed) if net_out_assumed > 0 else None
        poc["survival_days_assumed"] = (30.0 * hqla_total / net_out_assumed) if net_out_assumed > 0 else None

    state["poc_assumption"] = poc

    return state


def main():
    """
        CLI: python state_builder.py

        Loads:
            - portfolio_view.json: Aggregated positions/cashflows.
            - bank_profile.json: Liquidity/risk profile.
            - exchange_rate.json: FX map (optional; merges into portfolio).

        Writes:
            - state.json: Canonical output for downstream modules.

        Behavior:
            - Handles missing FX file gracefully.
            - Serializes datetimes as strings for JSON compatibility.
            - Prints success path; errors surface via build_state.

        PoC Notes:
            - Run after portfolio_aggregator.py.
            - For testing: Inspect state.json LCR/survival_days.
            - Extends to shocks via build_state(..., shocks={'rate': +200}).
        """
    base_dir = DATA_DIR
    portfolio_file = base_dir / "portfolio_view.json"
    bank_profile_file = base_dir / "bank_profile.json"
    fx_file = base_dir / "exchange_rate.json"
    out_file = base_dir / "state.json"

    with open(portfolio_file, "r") as f:
        portfolio = json.load(f)
    with open(bank_profile_file, "r") as f:
        bank_profile = json.load(f)
    if fx_file.exists():
        with open(fx_file, "r") as f:
            fx_rates = json.load(f)
        portfolio["fx_rates"] = fx_rates

    state = build_state(portfolio, bank_profile, shocks=None)
    with open(out_file, "w") as f:
        json.dump(state, f, indent=2, default=str)
    print(f"[STATE_BUILDER] Saved state to {out_file}")

# CLI entrypoint writes file
# Chains workflow: loads inputs, builds/writes state.json for reproducibility.
if __name__ == "__main__":
    main()