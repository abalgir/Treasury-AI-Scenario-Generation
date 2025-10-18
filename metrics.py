#!/usr/bin/env python3
"""
metrics.py — PoC metrics aggregator (drop-in replacement).

- Computes baseline metrics from with_whom/data/state.json
  * LCR & survival (raw recompute from cashflows, gross fallback if net <0)
  * 1-day 99% VaR (model from state.risk_exposures / risk_vols / risk_corr)
    - falls back to state.profile_var.var_1d_99_usd if model VaR is invalid/≈0
  * worst_10bd_outflow over next 30 calendar days from state.cashflows
  * Per-currency LCR for ECB multi-currency compliance (HQLA / net outflows by ccy)

Outputs: with_whom/data/baseline_metrics.json

Non-invasive design:
- Matches validator's conservative LCR (gross if inflows exceed outflows).
- Per-ccy LCR computed from cashflows/HQLA, falls back to portfolio_intel.json or equal split.
- Keeps profile var and debug blocks for auditability.

Module Overview:
----------------
This module serves as a dedicated aggregator for baseline treasury metrics in the Treasury AI Proof-of-Concept (PoC).
It processes the canonical state.json output from state_builder.py to recompute and validate key liquidity and risk indicators,
ensuring alignment with Basel III standards. Designed as a drop-in replacement for ad-hoc metric computations, it emphasizes
auditability through debug notes and conservative fallbacks (e.g., gross outflows for LCR if net negative).

Key Features:
-------------
- Liquidity Metrics: Recomputes LCR and survival days from raw cashflows/HQLA, mirroring show_baseline.py logic.
- Market Risk: Quick VaR model using bucketed exposures (FX/IR/CR), volatilities, and correlations; falls back to profiled VaR.
- Outflow Stress: Identifies worst 10-business-day outflow window within 30-day horizon for operational liquidity planning.
- Multi-Currency: Per-ccy LCR for ECB compliance, allocating HQLA proportionally via exposures or intel cache.
- Output: JSON with metrics, debug audit trails, and console summary for immediate review.

Regulatory Alignment:
---------------------
- LCR/NSFR: Basel III (BCBS 238) conservative netting; per-ccy for CRR Article 412.
- VaR: 1-day 99% (z=2.326) per market risk rules; assumes USD-equivalent exposures.
- Outflows: Business-day rolling sums support daily liquidity monitoring (Fed SR 10-6).

Dependencies:
-------------
- Python 3.8+: pathlib, json, re, math, datetime, timedelta (standard library).
- numpy, pandas: Numerical computations and DataFrame aggregations.

Usage in Workflow:
------------------
- Post state_builder.py: python metrics.py → baseline_metrics.json.
- Integration: Feeds into stress_tester.py for before/after comparisons; Streamlit for dashboard metrics.
- CLI: Writes JSON; prints summary (LCR, survival, VaR, per-ccy) for quick validation.
- Extensibility: Add dnII_100bp_usd (placeholder) for interest rate sensitivity.

Error Handling:
---------------
- Safe floats/defaults prevent crashes on malformed state.
- Fallbacks: Zero VaR if no data; equal HQLA split if no exposures/intel.
- Debug: Notes on sources/methods (e.g., "used gross outflows due to negative net").

Author:
-------
FintekMinds TreasuryAI PoC — Baseline Metrics Computation Module
Version: 2025-10 (Enhanced with per-ccy LCR and VaR modeling)
"""

from pathlib import Path
import json
import re
from math import sqrt
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Define file paths for input and output
BASE = Path(__file__).resolve().parent / "data"
STATE_FILE = BASE / "state.json"
OUT_FILE = BASE / "baseline_metrics.json"
PORTFOLIO_INTEL_FILE = BASE / "portfolio_intel.json"  # For HQLA fallback

# Z-score for 99% confidence level (one-tailed normal distribution)
Z99 = 2.326347874


def safe_float(x, default=0.0):
    """
    Safely coerce input to float with default fallback.

    Args:
        x: Any input (str, int, float, None).
        default: Value to return on failure (default: 0.0).

    Returns:
        float: Parsed value or default.

    Purpose:
        Handles malformed numerics in state.json (e.g., strings, None) for robust metric computation.
    """
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def compute_bucket_sigma_from_exposures(exposures, vols, bucket_key):
    """
    Compute standard deviation for a risk bucket (FX/IR/CR) using exposures and vols.

    Args:
        exposures: Nested dict of instrument exposures by bucket (e.g., state.risk_exposures['FX']).
        vols: Dict of volatilities by instrument (e.g., state.risk_vols['FX']).
        bucket_key: Str identifier ('FX', 'IR', 'CR') for sensitivity type.

    Returns:
        float: Bucket-level dollar standard deviation (sum of squared individual SDs).

    Process:
        - FX: |pos_usd| * vol_pct (relative volatility).
        - IR: |pv01_usd| * vol (parallel shift vol on PV01 scale).
        - CR: |cs01_usd| * vol (credit spread vol).
        - Aggregates via sqrt(sum(sd^2)) for undiversified bucket risk.

    Notes:
        - Assumes USD-equivalent; zero on empty/malformed inputs.
        - Used in portfolio VaR for bucket marginal contributions.
    """
    # Return 0.0 for empty or invalid exposures
    if not isinstance(exposures, dict) or not exposures:
        return 0.0
    vals = []
    for inst_key, inst in exposures.items():
        if not isinstance(inst, dict):
            continue
        # FX: Calculate dollar SD as absolute position times volatility
        if bucket_key == "FX":
            pos = safe_float(inst.get("pos_usd", inst.get("notional", inst.get("size", 0.0))))
            vol = safe_float((vols or {}).get(inst_key) or (vols or {}).get(inst_key.upper()) or 0.0)
            dollar_sd = abs(pos) * vol
            vals.append(dollar_sd)
        # IR: Calculate dollar SD as absolute PV01 times volatility
        elif bucket_key == "IR":
            pv01 = safe_float(inst.get("pv01_usd") or inst.get("pv01") or 0.0)
            vol = safe_float((vols or {}).get(inst_key) or 0.0)
            dollar_sd = abs(pv01) * vol
            vals.append(dollar_sd)
        # CR: Calculate dollar SD as absolute CS01 times volatility
        elif bucket_key == "CR":
            cs01 = safe_float(inst.get("cs01_usd") or inst.get("cs01") or 0.0)
            vol = safe_float((vols or {}).get(inst_key) or 0.0)
            dollar_sd = abs(cs01) * vol
            vals.append(dollar_sd)
    # Return 0.0 if no valid values computed
    if not vals:
        return 0.0
    # Aggregate SDs using square root of sum of squares
    sq = sum([v * v for v in vals])
    return sqrt(sq)


def compute_quick_var(state):
    """
    Quick 1-day 99% VaR computation using bucketed risks and correlations.

    Args:
        state: Dict from state.json with risk_exposures, risk_vols, risk_corr, profile_var.

    Returns:
        Dict: VaR details including sigmas, matrix, SD, final VaR (z=2.326), source/notes.

    Process:
        1. Compute bucket SDs (FX/IR/CR) via compute_bucket_sigma_from_exposures.
        2. Correlation: Provided 3x3 matrix or identity fallback.
        3. Portfolio variance: sigma_vec^T * C * sigma_vec; SD = sqrt(var).
        4. VaR = z99 * SD; fallback to profile_var if computed <=0/NaN.
        5. Audit: Notes on source (model/fallback/none), profile comparison.

    Compliance:
        - Aligns with Basel market risk VaR (historical/parametric); 1-day for daily monitoring.
        - Undiversified buckets; assumes no higher-order Greeks (e.g., convexity).

    Notes:
        - Z99=2.326 for 99% one-tailed normal.
        - Zero exposures/vols → zero VaR; enables graceful degradation.
    """
    # Extract risk data from state
    exposures = state.get("risk_exposures") or {}
    vols = state.get("risk_vols") or {}
    corr = state.get("risk_corr")

    # Compute standard deviations for each risk bucket
    sigma_fx = compute_bucket_sigma_from_exposures(exposures.get("FX", {}), vols.get("FX", {}), "FX")
    sigma_ir = compute_bucket_sigma_from_exposures(exposures.get("IR", {}), vols.get("IR", {}), "IR")
    sigma_cr = compute_bucket_sigma_from_exposures(exposures.get("CR", {}), vols.get("CR", {}), "CR")

    # Create sigma vector for FX, IR, CR
    sigma_vec = np.array([sigma_fx, sigma_ir, sigma_cr], dtype=float)

    # Handle correlation matrix: use provided or fallback to identity
    if corr and isinstance(corr, (list, tuple)) and len(corr) >= 3:
        try:
            C = np.array(corr, dtype=float)
            if C.shape != (3, 3):
                raise ValueError("risk_corr is not 3x3")
        except Exception:
            C = np.eye(3)
            corr_note = "invalid risk_corr provided; using identity matrix"
        else:
            corr_note = "using provided risk_corr"
    else:
        C = np.eye(3)
        corr_note = "no risk_corr provided; using identity matrix"

    # Compute portfolio standard deviation
    try:
        port_var = float(sigma_vec.dot(C).dot(sigma_vec))
        port_var = max(0.0, port_var)
        port_sd = sqrt(port_var)
    except Exception:
        port_sd = 0.0

    # Calculate VaR using z-score
    computed_var = float(Z99 * port_sd)

    # Extract profile VaR for fallback
    profile_var = (state.get("profile_var") or {}).get("var_1d_99_usd")
    profile_var_num = safe_float(profile_var, default=None)

    # Initialize VaR selection logic
    var_source = "model"
    final_var = computed_var
    notes = []

    # Fallback to profile VaR if computed is invalid
    if computed_var <= 0 or np.isnan(computed_var):
        if profile_var_num is not None and profile_var_num > 0:
            final_var = profile_var_num
            var_source = "profile_fallback"
            notes.append("computed_model_var_zero -> using profile_var as fallback")
        else:
            final_var = 0.0
            var_source = "none"
            notes.append("no computed nor profile var available")
    else:
        # Compare with profile VaR for audit if available
        if profile_var_num is not None:
            notes.append(f"profile_var_available (profile={profile_var_num:.2f}; computed={computed_var:.2f})")

    return {
        "sigma_vec": [float(x) for x in sigma_vec.tolist()],
        "corr_matrix": C.tolist(),
        "portfolio_sd": float(port_sd),
        "computed_var_1d_99": float(computed_var),
        "VaR_1d_99_usd": float(final_var),
        "_var_source": var_source,
        "_notes": notes
    }


def compute_worst_10bd_outflow(state):
    """
    Aggregates cashflows into business days (Mon-Fri) for the next 30 calendar days
    starting at state['as_of'] and finds the minimum rolling sum over 10 business days.
    Returns a dict with worst_10bd_outflow (negative or 0).

    Compute the most severe 10-business-day outflow window within 30-day horizon.

    Args:
        state: Dict from state.json with cashflows and as_of.

    Returns:
        Dict: {'worst_10bd_outflow': float (negative/0), 'as_of': str, 'period_start/end': str, 'message': str (if no data)}.

    Process:
        1. Filter cashflows to [as_of, as_of+30d]; aggregate daily sums.
        2. Reindex to business days (pd.bdate_range: Mon-Fri).
        3. Rolling sum (window=10, min_periods=1); min value (outflow focus: negative).
        4. Clip to 0 if positive (no stress); message on empty horizon.

    Purpose:
        - Identifies peak liquidity drawdown periods for operational planning (e.g., funding gaps).
        - Aligns with Fed/ ECB intraday/ short-term liquidity stress tests.

    Notes:
        - Assumes signed amounts (+in/-out); totals outflows if negative.
        - Holidays/weekends: bdate_range excludes; fill_value=0.
    """
    cfs = state.get("cashflows") or []
    as_of = state.get("as_of")
    if not as_of:
        as_of = pd.to_datetime("today").date().isoformat()
    as_of_dt = pd.to_datetime(as_of).date()
    end_dt = as_of_dt + timedelta(days=30)

    # Build cashflow rows
    rows = []
    for cf in cfs:
        d = cf.get("date") or cf.get("value_date") or as_of
        try:
            dd = pd.to_datetime(d).date()
        except Exception:
            continue
        if not (as_of_dt <= dd <= end_dt):
            continue
        amt = safe_float(cf.get("amount", 0.0))
        rows.append({"date": dd, "amount": amt})

    # Return empty result if no valid cashflows
    if not rows:
        return {"worst_10bd_outflow": 0.0, "message": "no cashflows in 30-day horizon"}

    # Aggregate to daily sums and reindex to business days
    df = pd.DataFrame(rows)
    df = df.groupby("date", as_index=True).sum().sort_index()
    bdays = pd.bdate_range(start=as_of_dt, end=end_dt)
    df = df.reindex(bdays, fill_value=0.0)
    s = df["amount"]
    # Compute rolling 10-business-day sum
    roll10 = s.rolling(window=10, min_periods=1).sum()
    worst = float(roll10.min())
    worst_outflow = worst if worst < 0 else 0.0
    return {"worst_10bd_outflow": worst_outflow, "as_of": as_of, "period_start": as_of_dt.isoformat(), "period_end": end_dt.isoformat()}


def _parse_date(date_str: str) -> datetime:
    """
    Parse date str to datetime; fallback to today.

    Args:
        date_str: ISO or similar string.

    Returns:
        datetime: Parsed or current.

    Purpose:
        Robust date handling for cashflow bucketing in LCR/survival.
    """
    try:
        return pd.to_datetime(date_str).to_pydatetime()
    except Exception:
        return datetime.now()


def _flows_30d(state):
    """
    Compute inflows/outflows from state['cashflows'] for 30d horizon (validator-style).

    Args:
        state: Dict from state.json.

    Returns:
        tuple[float, float]: (inflows, outflows) USD over [as_of, as_of+30d].

    Process:
        - Filters dates; signs by direction/amount (out/pay/debit/- → outflow abs; else inflow abs).
        - Mirrors show_baseline.py for consistency.

    Notes:
        - USD assumed; multi-ccy in per_ccy variant.
    """
    inflow = outflow = 0.0
    as_of = _parse_date(state.get("as_of") or pd.to_datetime("today").date().isoformat())
    end = as_of + timedelta(days=30)
    for cf in state.get("cashflows", []):
        try:
            dt_raw = cf.get("date") or cf.get("value_date") or as_of
            dt = _parse_date(dt_raw)
        except Exception:
            dt = as_of
        if not (as_of <= dt <= end):
            continue
        amt = safe_float(cf.get("amount", 0.0))
        dirraw = str(cf.get("direction", "")).lower()
        if dirraw in ("out", "pay", "debit", "payout", "-1", "-") or amt < 0:
            outflow += abs(amt)
        else:
            inflow += abs(amt)
    return inflow, outflow


def _hqla_total(state):
    """
    Compute total HQLA (USD equivalent) from state['hqla'].

    Args:
        state: Dict from state.json.

    Returns:
        float: Sum of amount_usd/amount.

    Notes:
        - Raw pre-haircut; effective in stressed.
    """
    hqla = state.get("hqla", []) or []
    total = 0.0
    for h in hqla:
        total += safe_float(h.get("amount_usd", h.get("amount", 0.0)))
    return total


def compute_per_ccy_lcr(state):
    """
    Compute per-currency LCR (HQLA / net outflows) for ECB multi-currency compliance.
    Allocates HQLA by state.exposures; falls back to portfolio_intel.json or equal split.
    Uses gross outflows if net <= 0, per Basel conservative approach.

    Multi-currency LCR computation with HQLA allocation.

    Args:
        state: Dict from state.json.

    Returns:
        Dict: {ccy: float LCR or 'N/A', '_debug': [notes]} for major ccys (USD/EUR/GBP + active).

    Process:
        1. Aggregate cashflows by ccy/direction over 30d; compute inflows/outflows.
        2. Net = max(0, outflows - inflows); denom = outflows if net<=0 else net.
        3. HQLA alloc: Proportional to exposures; fallback parse portfolio_intel or equal split.
        4. LCR = alloc_hqla / denom; 'N/A' on zero denom.
        5. Debug: Allocation method, gross usage, parsing notes.

    Compliance:
        - CRR Article 412: Currency-specific >=100% where >5% of total outflows.
        - Conservative: Gross fallback prevents inflated ratios.

    Notes:
        - Defaults to USD; handles unknown directions as inflows if positive.
        - Intel parse: Regex for "Level X: Y.B (CCY Z%)" → weighted exposure.
    """
    cfs = state.get("cashflows", []) or []
    hqla = state.get("hqla", []) or []
    as_of = state.get("as_of") or pd.to_datetime("today").date().isoformat()
    as_of_dt = pd.to_datetime(as_of).date()
    end_dt = as_of_dt + timedelta(days=30)

    # Compute total HQLA (USD equivalent)
    total_hqla = sum(safe_float(h.get("amount_usd", h.get("amount", 0.0))) for h in hqla)
    if total_hqla <= 0:
        return {"USD": "N/A", "EUR": "N/A", "GBP": "N/A", "_debug": ["no HQLA available"]}

    # Aggregate cashflows by currency
    cf_rows = []
    for cf in cfs:
        date = cf.get("date") or cf.get("value_date") or as_of
        try:
            dd = pd.to_datetime(date).date()
        except Exception:
            continue
        if not (as_of_dt <= dd <= end_dt):
            continue
        amt = safe_float(cf.get("amount", 0.0))
        ccy = str(cf.get("currency", "USD")).upper()
        direction = str(cf.get("direction", "unknown")).lower()
        cf_rows.append({"ccy": ccy, "amount": amt, "direction": direction})

    if not cf_rows:
        return {"USD": "N/A", "EUR": "N/A", "GBP": "N/A", "_debug": ["no cashflows in 30d horizon"]}

    df = pd.DataFrame(cf_rows)
    agg_flows = df.groupby(["ccy", "direction"])["amount"].sum().unstack(fill_value=0)
    agg_flows["inflows"] = agg_flows.get("in", 0) + agg_flows.get("unknown", 0) * (agg_flows.get("unknown", 0) > 0)
    agg_flows["outflows"] = abs(agg_flows.get("out", 0) + agg_flows.get("unknown", 0) * (agg_flows.get("unknown", 0) < 0))

    # Allocate HQLA by ccy exposure (from state.exposures or portfolio_intel.json)
    exp_by_ccy = {}
    exposures = state.get("exposures", []) or []
    for exp in exposures:
        ccy = str(exp.get("currency", "USD")).upper()
        notional = safe_float(exp.get("notional", exp.get("amount", 0.0)))
        exp_by_ccy[ccy] = exp_by_ccy.get(ccy, 0.0) + abs(notional)

    total_exp = sum(exp_by_ccy.values())
    debug_notes = []
    if total_exp <= 0:
        # Fallback to portfolio_intel.json hql_by_level
        try:
            with open(PORTFOLIO_INTEL_FILE, "r") as f:
                intel = json.load(f)
            hql_str = intel.get("hql_by_level", "")
            if hql_str:
                # Parse e.g., "Level 1: 3.0B (EUR 52%); Level 2A: 0.6B (USD 61%)"
                for part in hql_str.split(";"):
                    part = part.strip()
                    if not part:
                        continue
                    match = re.match(r"(\w+):\s*(\d+\.\d)B\s*\((\w+)\s*(\d+)%\)", part)
                    if match:
                        level, total_b, ccy, pct = match.groups()
                        total_usd = float(total_b) * 1e9 * (float(pct) / 100)  # B to USD, weighted by %
                        exp_by_ccy[ccy] = exp_by_ccy.get(ccy, 0.0) + total_usd
                        debug_notes.append(f"parsed {level} for {ccy}: {total_b}B at {pct}%")
            total_exp = sum(exp_by_ccy.values()) or 1.0
        except Exception:
            # Last resort: Equal split across active ccys
            active_ccys = set(agg_flows.index) | {"USD", "EUR", "GBP"}
            for ccy in active_ccys:
                exp_by_ccy[ccy] = total_hqla / len(active_ccys)
            total_exp = sum(exp_by_ccy.values()) or 1.0
            debug_notes.append("fallback: equal HQLA split")

    per_ccy = {}
    for ccy in set(agg_flows.index) | set(exp_by_ccy.keys()):
        inflows = agg_flows.get("inflows", {}).get(ccy, 0.0)
        outflows = agg_flows.get("outflows", {}).get(ccy, 0.0)
        net_outflows = max(0.0, outflows - inflows)  # Clip per Basel
        # Use gross outflows if net <= 0, per Basel conservative approach
        denominator = outflows if net_outflows <= 0 and outflows > 0 else net_outflows
        if denominator <= 0:
            per_ccy[ccy] = "N/A"
            debug_notes.append(f"no valid denominator for {ccy} (net={net_outflows}, outflows={outflows})")
            continue
        # Allocate HQLA proportionally to exposure
        ccy_weight = exp_by_ccy.get(ccy, 0.0) / total_exp
        hqla_alloc = total_hqla * ccy_weight if total_exp > 0 else total_hqla / len(set(agg_flows.index) | {ccy})
        per_ccy[ccy] = hqla_alloc / denominator if denominator > 0 else "N/A"
        if net_outflows <= 0 and outflows > 0:
            debug_notes.append(f"used gross outflows for {ccy} due to negative net")

    if debug_notes:
        per_ccy["_debug"] = debug_notes

    return per_ccy


def build_baseline_metrics(state):
    """
    Aggregate all baseline metrics into structured dict.

    Args:
        state: Dict from state.json.

    Returns:
        Dict: {'as_of': str, 'baseline_metrics': {LCR, per_currency_LCR, survival_days, VaR_1d_99, worst_10bd_outflow, dnII_100bp_usd:0.0, breaches:[], 'debug': {...}}}.

    Structure:
        - Liquidity: LCR/survival with raw audit (inflows/outflows/net/method).
        - Risk: VaR with computed/profile/source notes.
        - Stress: Worst 10bd outflow with period details.
        - Per-ccy: Dict of LCRs + debug allocation notes.
        - Debug: Nested for var (source/computed) and lcr (raws/method).

    Notes:
        - Breaches/dnII placeholder for future (compliance/IRS sensitivity).
        - Ensures None → float(0)/None handling for JSON safety.
    """
    # Recompute raw LCR/survival (validator-style)
    hqla = _hqla_total(state)
    inflows, outflows = _flows_30d(state)
    raw_net = outflows - inflows
    net_clipped = max(0.0, raw_net)

    # Calculate LCR and survival days, with gross fallback if net outflows <= 0
    lcr = hqla / net_clipped if net_clipped > 0 else None
    survival_days = (30.0 * hqla / net_clipped) if net_clipped > 0 else None
    lcr_gross = hqla / outflows if outflows > 0 else None
    survival_gross = (30.0 * hqla / outflows) if outflows > 0 else None

    # Audit LCR computation details
    debug_lcr = {
        "raw_inflows_usd": inflows,
        "raw_outflows_usd": outflows,
        "raw_net_outflows": raw_net,
        "method": "net"
    }
    if raw_net <= 0:
        debug_lcr["method"] = "gross (inflows exceed outflows)"
        lcr = lcr_gross
        survival_days = survival_gross

    # Compute per-currency LCR
    per_ccy = compute_per_ccy_lcr(state)

    # Compute VaR
    var_info = compute_quick_var(state)
    # Compute worst 10-business-day outflow
    w10 = compute_worst_10bd_outflow(state)

    # Construct metrics dictionary
    metrics = {
        "as_of": state.get("as_of"),
        "baseline_metrics": {
            "LCR": float(lcr) if lcr is not None else None,
            "per_currency_LCR": per_ccy,  # Updated: Computed per-ccy LCR
            "survival_days": float(survival_days) if survival_days is not None else None,
            "VaR_1d_99": float(var_info["VaR_1d_99_usd"]),
            "worst_10bd_outflow": float(w10["worst_10bd_outflow"]),
            "dnII_100bp_usd": 0.0,  # Placeholder for interest rate sensitivity
            "breaches": [],  # Placeholder for compliance violations
            "debug": {
                "var": {
                    "computed_var_1d_99": float(var_info["computed_var_1d_99"]),
                    "profile_var_1d_99": safe_float((state.get("profile_var") or {}).get("var_1d_99_usd")),
                    "_var_source": var_info.get("_var_source"),
                    "_notes": var_info.get("_notes", [])
                },
                "lcr": debug_lcr,  # Updated: Raw recompute audit
                "survival_summary": {"raw_net_clipped": net_clipped}
            }
        }
    }
    return metrics


def main():
    """
    CLI Entry: Load state, compute metrics, write JSON, print summary.

    Behavior:
        - Requires state.json; exits on missing.
        - Writes indented baseline_metrics.json.
        - Console: Key metrics (LCR/survival/VaR/outflow) + per-ccy LCR.

    PoC Usage:
        - Automate in orchestrator: metrics = build_baseline_metrics(state); json.dump(...).
        - For Streamlit: Load JSON → st.metric('LCR', bm['LCR']); st.text(per_ccy).
    """
    # Check for state file existence
    if not STATE_FILE.exists():
        raise SystemExit(f"State file not found: {STATE_FILE}")
    # Load state.json
    state = json.loads(STATE_FILE.read_text())

    # Compute metrics
    metrics = build_baseline_metrics(state)

    # Write metrics to JSON file
    OUT_FILE.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"[metrics] Wrote baseline metrics to {OUT_FILE}")
    # Print summary to console
    bm = metrics["baseline_metrics"]
    print(f"[metrics] LCR={bm['LCR']} survival_days={bm['survival_days']} VaR_1d_99={bm['VaR_1d_99']:.2f} worst_10bd_outflow={bm['worst_10bd_outflow']}")
    # Print per-currency LCR for visibility
    per_ccy = bm.get("per_currency_LCR", {})
    if per_ccy:
        print(f"[metrics] Per-ccy LCR: {', '.join(f'{ccy}:{v:.2f}' if isinstance(v, (int, float)) else f'{ccy}:{v}' for ccy, v in per_ccy.items())}")


if __name__ == "__main__":
    main()