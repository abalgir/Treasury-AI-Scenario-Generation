#!/usr/bin/env python3
"""
macro_market_data.py — Fetches/aggregates macro data for AI scenarios.

Purpose:
- PoC: Fake Oct 2025 values (Fed/ECB projections: CPI 2.9%, 10y 4.05%, GDP 1.3%).
- Live stub: For web_search/tools (e.g., "US CPI October 2025").
- Cache to macro_data.json (<24h fresh? load; else recompute/save).
- Output: Dict with keys like "cpi_us": 2.9, "treasury_10y_us": 4.05 for macro_inputs.

Non-invasive: Callable from orchestrator_phase1.py; real-life accessible via FMP/Fred/ECB APIs.

Module Overview:
----------------
This module aggregates macroeconomic and market data essential for generating stress scenarios in the Treasury AI Proof-of-Concept (PoC).
It provides plausible forward-looking values for October 2025 based on Fed/ECB/IMF projections, serving as a stub for live data ingestion.
The output dict feeds into scenario generation (e.g., rate shocks from treasury curves) and risk computations (e.g., inflation-linked VaR adjustments).
Caching ensures efficiency in iterative PoC runs, with a 24-hour TTL to balance freshness and compute overhead.

Key Features:
-------------
- Fake Data Mode: Hardcoded projections for US/EU indicators (treasury rates, GDP, CPI, unemployment, etc.) tailored to Oct 2025.
- Caching: JSON persistence with staleness check; recomputes only if >24h old or invalid.
- Extensibility: Live stub raises NotImplementedError—integrate with web_search or APIs (FMP, FRED, ECB) for production.
- Treasury Focus: Covers liquidity-relevant metrics (e.g., federal funds for LCR outflows) and ALM inputs (e.g., yield curve for duration gaps).

Regulatory Alignment:
---------------------
- Supports Basel III/NSFR scenario testing via GDP/inflation shocks (e.g., +50bps CPI → behavioral outflows).
- Yield curve data aligns with ECB/Fed stress guidelines (e.g., +200bps parallel shift).
- Ensures forward-looking estimates for CRD IV Pillar 2 ICAAP (Internal Capital Adequacy Assessment Process).

Dependencies:
-------------
- Python 3.8+: pathlib, typing, json, time, datetime (standard library).
- No external libs; pure Python for PoC portability.

Usage in Workflow:
------------------
- Upstream: Called in orchestrator_phase1.py before scenario proposal.
- This: aggregate_macro_data() → macro_data.json (dict for prompts: "Apply +100bps to 10y treasury at 4.05%").
- Downstream: Feeds stress_tester.py (e.g., shock treasury_10y_us); Streamlit for dashboard (e.g., st.metric('10y Yield', macro['us_treasury_rates']['year10'])).
- CLI: python macro_market_data.py (dumps JSON; uses fake mode).
- Production: Set use_fake=False; implement fetch_live_data with tools (e.g., web_search "US CPI October 2025 site:bls.gov").

Configuration Notes:
--------------------
- as_of_date: ISO 'YYYY-MM-DD' (default '2025-10-15'); contextualizes data.
- use_fake: True for PoC (default); False triggers live stub (extend with FRED/ECB APIs).
- Cache: macro_data.json; mtime-based TTL; overwrites on recompute.
- Data Sources (Fake): Derived from Fed SEP (Sep 2025), ECB projections, IMF WEO—plausible, not real-time.

Error Handling:
---------------
- Missing cache: Recomputes gracefully.
- Invalid cache: Prints warning, falls back to fresh compute.
- Live Mode: Raises NotImplementedError to enforce PoC boundaries.

Author:
-------
FintekMinds TreasuryAI PoC — Macro Data Aggregation Module
Version: 2025-10 (Fake Projections for Scenario Shocks)
"""

from pathlib import Path
from typing import Dict, Any
import json
import time
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = DATA_DIR / "macro_data.json"
STALENESS_HOURS = 24


def _file_age_hours(p: Path) -> float | None:
    """
    Compute file age in hours from modification time.

    Args:
        p: Path to check (e.g., CACHE_FILE).

    Returns:
        float or None: Age in hours if exists, else None.

    Purpose:
        Determines cache freshness for macro data; triggers recompute if stale.
    """
    if not p.exists():
        return None
    return max(0.0, (time.time() - p.stat().st_mtime) / 3600.0)


def fetch_fake_data(as_of_date: str = "2025-10-15") -> Dict[str, Any]:
    """
    PoC fake: Plausible Oct 2025 values from Fed/ECB/IMF projections.
    Covers treasury rates, GDP, CPI, unemployment, etc. for shocks (rates_bps, vol proxies).

    Generate hardcoded macroeconomic data for PoC scenario generation.

    Args:
        as_of_date: ISO str for data context (default: '2025-10-15').

    Returns:
        Dict[str, Any]: Nested indicators (e.g., {'us_treasury_rates': {'year10': 4.05}, 'cpi_us': 2.9}).

    Structure:
        - us_treasury_rates: Yield curve (month1 to year30) in %.
        - gdp_growth_us/real_gdp_us: Q3 2025 forecast (% and trillions USD).
        - cpi_us/inflation_rate_us: Aug-Oct est. (%).
        - unemployment_rate_us: Sep est. (%).
        - federal_funds_us: Target range high (%).
        - retail_sales_us/industrial_production_us: MoM %.
        - ecb_deposit_rate: Oct 2025 est. (%).
        - euribor_6m: 6M rate (%).
        - gdp_growth_eu/cpi_eu: Eurozone Q3/Sep est. (%).

    Notes:
        - Derived from Fed SEP (Sep 2025), ECB staff projections, IMF WEO—illustrative only.
        - Used for shocks: e.g., +200bps to year10 for rate stress; +1% CPI for inflation behavioral.
        - Treasury Relevance: Yield curve for DV01/PV01; GDP/CPI for LCR runoff assumptions.
    """
    return {
        "date": as_of_date,
        "us_treasury_rates": {
            "month1": 5.53,
            "month2": 5.5,
            "month3": 5.45,
            "month6": 5.3,
            "year1": 5.01,
            "year2": 4.64,
            "year3": 4.43,
            "year5": 4.26,
            "year7": 4.28,
            "year10": 4.05,  # Adjusted for Oct 2025 est.
            "year20": 4.51,
            "year30": 4.38
        },
        "gdp_growth_us": 1.3,  # Q3 2025 forecast
        "real_gdp_us": 28.5,  # Trillions USD
        "cpi_us": 2.9,  # Aug-Oct est.
        "inflation_rate_us": 2.9,
        "unemployment_rate_us": 4.2,  # Sep est.
        "federal_funds_us": 4.75,  # Target range high
        "retail_sales_us": 0.5,  # MoM %
        "industrial_production_us": 0.3,  # MoM %
        "ecb_deposit_rate": 3.25,  # Oct 2025 est.
        "euribor_6m": 2.1,
        "gdp_growth_eu": 0.8,  # Eurozone Q3 est.
        "cpi_eu": 1.8  # Eurozone Sep est.
    }


def fetch_live_data(as_of_date: str = "2025-10-15") -> Dict[str, Any]:
    """
    Live stub: For tools (e.g., web_search "US CPI October 2025").
    Raise NotImplemented for PoC—implement with tools for production.

    Placeholder for real-time macro data fetch.

    Args:
        as_of_date: ISO str for query context.

    Returns:
        Dict[str, Any]: Live data (not implemented).

    Raises:
        NotImplementedError: Enforces PoC use of fake data; extend with web_search/FRED/ECB APIs.

    Production Extension:
        - Use tools: web_search(query=f"US CPI {as_of_date} site:bls.gov", num_results=1).
        - Parse results for values (e.g., regex for "%").
        - Fallback to latest available if as_of_date future.
    """
    raise NotImplementedError("Live fetch via web_search; stub for PoC")


def aggregate_macro_data(
    as_of_date: str = "2025-10-15",
    use_fake: bool = True  # PoC flag: True=fake, False=live
) -> Dict[str, Any]:
    """
    Main func: Cache-checked aggregate of macro data.

    Args:
        as_of_date: ISO date for context (default: '2025-10-15').
        use_fake: PoC mode (default True).

    Returns:
        Dict: As fetch_fake/live, loaded from cache if fresh (<24h).

    Process:
        1. Check cache age: If <24h and valid JSON, load/print age.
        2. Else: Fetch (fake/live), add computed_at, save indented JSON.
        3. Prints status for CLI/debug.

    Cache Management:
        - TTL: 24 hours from mtime.
        - Invalid: Recompute on parse fail.
        - Non-invasive: No overwrite if fresh.

    Treasury Integration:
        - Output keys for prompts: e.g., "Shock 10y US Treasury from {macro['us_treasury_rates']['year10']}% +200bps."
        - Compliance: Ensures scenario realism (e.g., ECB deposit rate for EUR outflows).
    """
    age = _file_age_hours(CACHE_FILE)
    if age is not None and age < STALENESS_HOURS:
        try:
            cached = json.loads(CACHE_FILE.read_text())
            print(f"[macro_data] Loaded fresh cache (age={age:.1f}h)")
            return cached
        except Exception:
            print("[macro_data] Cache invalid; recomputing")

    print("[macro_data] Computing fresh data...")
    if use_fake:
        data = fetch_fake_data(as_of_date)
    else:
        data = fetch_live_data(as_of_date)
    data["computed_at"] = datetime.now().isoformat()
    CACHE_FILE.write_text(json.dumps(data, indent=2, default=str))
    print(f"[macro_data] Saved to {CACHE_FILE}")
    return data


if __name__ == "__main__":
    """
    CLI test: Run aggregator (PoC fake mode)
    - Calls aggregate_macro_data(use_fake=True) with default as_of.
    - Dumps full JSON to stdout (indented, str for datetimes).
    - Use: python macro_market_data.py > test_macro.json for validation.

    PoC Usage:
        - Integrates with orchestrator: macro = aggregate_macro_data(); prompt = f"Generate shocks based on CPI {macro['cpi_us']}%".
        - For Streamlit: st.json(macro) or st.metric('US 10y Yield', f"{macro['us_treasury_rates']['year10']}%").
    """
    # CLI test: Run aggregator (PoC fake mode)
    macro = aggregate_macro_data(use_fake=True)
    print(json.dumps(macro, indent=2, default=str))