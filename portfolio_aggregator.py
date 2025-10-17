# portfolio_aggregator.py
"""
Portfolio Aggregator for the Treasury AI Proof-of-Concept (PoC)

Purpose:
--------
Aggregates all relevant treasury data (counterparty flows, bank profile,
FX rates) into a single, normalized portfolio view for downstream
stress testing, scenario building, and dashboard visualization.

Output:
--------
- with_whom/data/portfolio_view.json  ← consolidated baseline snapshot

Reads:
-------
- with_whom/data/counterparty_data.json   : core counterparty datasets (cashflows, limits, etc.)
- with_whom/data/bank_profile.json        : structural bank-level liquidity profile
- with_whom/data/exchange_rate.json       : optional FX conversion table (for USD normalization)

Author:
--------
FintekMinds TreasuryAI — Proof-of-Concept module
Version: 2025-10
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
import statistics


# ===============================================================
# --- Configuration and paths
# ===============================================================
# Define the canonical data directory for this module
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp used for generated records
TODAY = datetime.utcnow()


# ===============================================================
# --- I/O HELPERS
# ===============================================================
def load_json(name: str) -> Any:
    """
    Load a JSON file from DATA_DIR.

    Args:
        name: filename (e.g., 'counterparty_data.json')
    Returns:
        Parsed Python object.
    Raises:
        FileNotFoundError if the file is missing.
    """
    p = DATA_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    return json.loads(p.read_text())


def write_json(name: str, payload: Any) -> None:
    """
    Write a dictionary to a JSON file in DATA_DIR, pretty-printed.

    Args:
        name: filename for output.
        payload: any JSON-serializable object.
    """
    (DATA_DIR / name).write_text(json.dumps(payload, indent=2, default=str))


# ===============================================================
# --- FX HELPERS
# ===============================================================
def _load_fx_rates() -> Dict[str, float]:
    """
    Load optional FX rates from exchange_rate.json.

    Returns:
        Mapping {currency_code: USD_rate}. If file not found or invalid,
        returns empty dict (defaults to 1.0 multiplier downstream).
    """
    fx_path = DATA_DIR / "exchange_rate.json"
    if fx_path.exists():
        try:
            return json.loads(fx_path.read_text())
        except Exception:
            return {}
    return {}


def _usd(amount: float, ccy: str, fx_map: Dict[str, float]) -> float:
    """
    Convert an amount in a given currency to USD using provided FX map.

    Args:
        amount: numeric amount in native currency.
        ccy: ISO currency code.
        fx_map: dict mapping currencies to USD conversion rates.

    Returns:
        Float value in USD (0.0 on error or missing rate).
    """
    try:
        rate = float(fx_map.get(str(ccy).upper(), 1.0) or 1.0)
        return float(amount or 0.0) * rate
    except Exception:
        return 0.0


# ===============================================================
# --- BANK PROFILE NORMALIZER
# ===============================================================
def normalize_bank_profile(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the bank_profile.json into canonical structure expected
    by downstream modules (state_builder, scenario_builder, etc.).

    Normalized keys:
        - intraday_liquidity: {reserve, currency}
        - hqla: list[{level, amount_usd, haircut, currency}]
        - limits, risk_exposures
        - max_survival_days
        - simulated_baseline (optional)
        - poc_inflow_fraction (from governance.poc.inflow_fraction_available_under_stress)

    This ensures heterogeneous bank profile shapes (from different data sources)
    map to one consistent schema.
    """
    bp = raw or {}
    normalized: Dict[str, Any] = {}

    # --- Intraday liquidity ---
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

    # --- HQLA breakdown normalization ---
    hqla_list = []
    if "hqla" in bp and isinstance(bp["hqla"], list):
        # Direct list form already compliant
        hqla_list = bp["hqla"]
    else:
        liquidity_block = bp.get("liquidity", {})
        breakdown = liquidity_block.get("hqla_breakdown") or {}
        if isinstance(breakdown, dict) and breakdown:
            # Convert nested dict form into list of dicts
            for lvl, info in breakdown.items():
                try:
                    amt = float(info.get("amount_usd", info.get("amount", 0.0)) or 0.0)
                except Exception:
                    amt = 0.0
                haircut = info.get("haircut", None)
                ccy = info.get("currency") or liquidity_block.get("currency") or "USD"
                hqla_list.append({"level": lvl, "amount_usd": amt, "haircut": haircut, "currency": ccy})
        # Fallback: single aggregate HQLA figure
        if not hqla_list and liquidity_block.get("total_hqla_usd") is not None:
            try:
                amt = float(liquidity_block.get("total_hqla_usd") or 0.0)
            except Exception:
                amt = 0.0
            hqla_list.append({"level": "aggregate", "amount_usd": amt, "haircut": None, "currency": "USD"})
    normalized["hqla"] = hqla_list

    # --- Limits & risk exposures ---
    normalized["limits"] = bp.get("limits", bp.get("counterparty_limits", {}))
    normalized["risk_exposures"] = bp.get("risk_exposures", bp.get("market_risk", {}))

    # --- Compute or inherit max_survival_days ---
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
                max_days = 30.0 * total_hqla / net_out if net_out > 0 else 120
            except Exception:
                max_days = 120
    normalized["max_survival_days"] = float(max_days or 120)

    # --- Preserve optional simulation metadata ---
    normalized["simulated_baseline"] = bp.get("simulated_baseline", {})
    normalized["_raw_profile_meta"] = {"meta": bp.get("meta", {}), "original_keys": list(bp.keys())}

    # --- Governance PoC inflow fraction ---
    governance = bp.get("governance", {}) or {}
    poc = governance.get("poc", {}) if isinstance(governance, dict) else {}
    normalized["poc_inflow_fraction"] = float(poc.get("inflow_fraction_available_under_stress", 0.20))

    return normalized


# ===============================================================
# --- SUMMARY HELPERS
# ===============================================================
def summarize_compliance(counterparties: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize limit compliance across all counterparties.
    Computes breaches and utilization statistics.
    """
    utilizations = []
    breaches = 0
    cp_summaries = []
    for cp in counterparties:
        comp = cp.get("compliance")
        if not comp:
            continue
        util = comp.get("limit_utilization")
        settlement = comp.get("settlement_limit", 0)
        mtm = comp.get("mtm_limit", 0)
        bank_limit = comp.get("bank_limit", 5_000_000)
        total_used = settlement + mtm
        breach = total_used > bank_limit

        if util is not None:
            utilizations.append(util)
        if breach:
            breaches += 1

        cp_summaries.append({
            "counterparty_id": cp.get("counterparty_id", "UNKNOWN"),
            "limit_utilization": util,
            "breach": breach
        })

    avg_util = (sum(utilizations) / len(utilizations)) if utilizations else None
    max_util = max(utilizations) if utilizations else None
    return {
        "cp_compliance": cp_summaries,
        "compliance_summary": {
            "total_counterparties": len(cp_summaries),
            "breaches": breaches,
            "avg_utilization": avg_util,
            "max_utilization": max_util
        }
    }


def summarize_settlement(counterparties: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize settlement performance statistics across counterparties:
    error rates, total transactions, and SWIFT mismatches.
    """
    error_rates = []
    total_tx = 0
    total_mismatches = 0
    high_error_cps = []
    for cp in counterparties:
        st = cp.get("settlement_trends")
        if not st:
            continue
        tx = st.get("total_transactions", 0)
        err = st.get("error_rate", 0.0)
        mism = st.get("swift_mismatches", 0)
        if tx > 0:
            error_rates.append(err)
            total_tx += tx
            total_mismatches += mism
            if err > 0.1:
                high_error_cps.append(cp.get("counterparty_id", "UNKNOWN"))
    return {
        "settlement_summary": {
            "avg_error_rate": (sum(error_rates) / len(error_rates)) if error_rates else None,
            "total_transactions": total_tx,
            "total_swift_mismatches": total_mismatches,
            "high_error_counterparties": high_error_cps
        }
    }


def summarize_payments(counterparties: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics over all historical payment records.
    Returns count, average, and median amount.
    """
    all_amounts = []
    for cp in counterparties:
        pays = cp.get("historical_payments", [])
        for p in pays:
            amt = p.get("amount")
            if amt is not None:
                all_amounts.append(amt)
    if all_amounts:
        avg_amt = statistics.mean(all_amounts)
        med_amt = statistics.median(all_amounts)
        count = len(all_amounts)
    else:
        avg_amt = med_amt = None
        count = 0
    return {"historical_payments_summary": {"count": count, "avg_amount": avg_amt, "median_amount": med_amt}}


# ===============================================================
# --- MAIN AGGREGATOR
# ===============================================================
def build_portfolio_view() -> Dict[str, Any]:
    """
    Build and persist the consolidated portfolio view.

    Pipeline:
        1. Load counterparty_data.json and bank_profile.json
        2. Normalize bank profile (HQLA, reserves, limits)
        3. Extract all exposures and cashflows from counterparties
        4. Add intraday reserve as pseudo-exposure & inflow
        5. Compute per-currency and USD totals (FX conversion)
        6. Add compliance, settlement, and payment summaries
        7. Persist to portfolio_view.json

    Returns:
        portfolio_view (dict) — canonical structure for state_builder & scenario_builder.
    """
    # --- Load inputs ---
    counterparties: List[Dict[str, Any]] = load_json("counterparty_data.json")
    raw_bank_profile: Dict[str, Any] = load_json("bank_profile.json")
    bank_profile = normalize_bank_profile(raw_bank_profile)

    all_cashflows: List[Dict[str, Any]] = []
    all_exposures: List[Dict[str, Any]] = []

    # --- Aggregate exposures and cashflows from counterparties ---
    for cp in counterparties:
        cp_id = cp.get("counterparty_id", "UNKNOWN")
        # Cashflows
        for cf in cp.get("cashflows", []):
            row = dict(cf)
            row["counterparty_id"] = cp_id
            if "instrument_id" not in row and "id" in row:
                row["instrument_id"] = row.pop("id")
            row.setdefault("product", cf.get("type"))
            all_cashflows.append(row)
        # Liquidity exposures
        for inst in cp.get("liquidity_profile", []):
            inst_copy = dict(inst)
            inst_copy["counterparty_id"] = cp_id
            if "instrument_id" not in inst_copy and "id" in inst_copy:
                inst_copy["instrument_id"] = inst_copy.pop("id")
            all_exposures.append(inst_copy)

    # --- Include intraday reserve as both exposure and inflow ---
    reserve = float(bank_profile.get("intraday_liquidity", {}).get("reserve", 0.0) or 0.0)
    reserve_ccy = bank_profile.get("intraday_liquidity", {}).get("currency", "USD")
    if reserve > 0:
        # Exposure entry
        reserve_inst = {
            "instrument_id": "CASH_RESERVE",
            "type": "cash",
            "currency": reserve_ccy,
            "notional": reserve,
            "maturity": TODAY.strftime("%Y-%m-%d"),
            "hql_level": "Level 1",
            "risk_driver": "none",
            "counterparty_id": "BANK"
        }
        all_exposures.append(reserve_inst)
        # Matching inflow entry
        reserve_cf = {
            "instrument_id": "CASH_RESERVE",
            "type": "cash",
            "currency": reserve_ccy,
            "date": TODAY.strftime("%Y-%m-%d"),
            "amount": reserve,
            "direction": "in",
            "description": "Initial cash reserve (from bank_profile)",
            "product": "cash",
            "counterparty_id": "BANK"
        }
        all_cashflows.append(reserve_cf)

    # --- Macro placeholder (can be expanded in scenario_builder) ---
    macro_inputs = {"gdp_growth": 1.2, "inflation": 2.5, "policy_rate": 3.0, "credit_spread": 1.0}

    # --- Aggregate totals and convert to USD ---
    fx = _load_fx_rates()
    per_ccy_sums: Dict[str, float] = {}
    gross_usd = 0.0
    net_usd = 0.0
    for r in all_exposures:
        ccy = str(r.get("currency", "USD")).upper()
        notional = r.get("notional") or r.get("amount") or r.get("face") or r.get("par") or 0.0
        try:
            n = float(notional)
        except Exception:
            n = 0.0
        per_ccy_sums[ccy] = per_ccy_sums.get(ccy, 0.0) + n
        usd_val = _usd(n, ccy, fx)
        net_usd += usd_val
        gross_usd += abs(usd_val)

    # --- Summary helpers ---
    compliance_data = summarize_compliance(counterparties)
    settlement_data = summarize_settlement(counterparties)
    payments_data = summarize_payments(counterparties)

    # --- Assemble the final view ---
    portfolio_view = {
        "as_of": TODAY.strftime("%Y-%m-%d"),
        "cashflows_preview": {"baseline": all_cashflows},
        "exposures": all_exposures,
        "macro_inputs": macro_inputs,
        "inputs": {"reserve_usd": reserve},
        "fx_rates": fx,
        "positions_by_currency": per_ccy_sums,
        "positions_gross_usd": round(gross_usd, 2),
        "total_notional_usd": round(gross_usd, 2),
        "bank_profile_summary": {
            "max_survival_days": bank_profile.get("max_survival_days"),
            "hqla_total_usd": sum(float(x.get("amount_usd", 0.0) or 0.0) for x in bank_profile.get("hqla", [])),
            "intraday_reserve_usd": bank_profile.get("intraday_liquidity", {}).get("reserve", 0.0)
        },
        **compliance_data,
        **settlement_data,
        **payments_data
    }

    # --- Persist and report ---
    write_json("portfolio_view.json", portfolio_view)
    print(
        f"Wrote portfolio_view.json with {len(all_exposures)} exposures "
        f"and {len(all_cashflows)} cashflows | gross_usd={portfolio_view['positions_gross_usd']:,}"
    )
    return portfolio_view


# ===============================================================
# --- CLI ENTRY POINT
# ===============================================================
if __name__ == "__main__":
    """
    Command-line execution:
        $ python portfolio_aggregator.py
    Generates: with_whom/data/portfolio_view.json
    """
    build_portfolio_view()
