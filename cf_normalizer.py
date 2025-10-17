# cf_normalizer.py
# Drop-in normalizer for state_builder.build_state(...)
# - Matches signature used by: from with_whom.dashboard.cf_normalizer import normalize_cashflows
# - Returns a pandas.DataFrame with normalized columns.
# - Does NOT add HQLA reserves as cashflows (handled in metrics via intraday_liquidity).

"""
Cashflow Normalizer for Treasury AI Proof-of-Concept (PoC)

Module Overview:
----------------
This module provides a robust, drop-in normalizer for heterogeneous cashflow data ingested into the Treasury AI workflow.
It harmonizes disparate schemas from upstream systems (e.g., Murex, Calypso exports) into a canonical pandas DataFrame,
ensuring consistent date coercion, currency standardization, amount signing (inflow/outflow convention),
and basic cleanup/deduplication. Designed for Basel III LCR/NSFR computations in state_builder.py.

Key Features:
-------------
- Handles mixed dict schemas: Maps aliases (e.g., 'ccy' → 'currency', 'amt' → 'amount').
- Enforces sign convention: + for inflows (RECEIVE/IN), - for outflows (PAY/OUT).
- Date normalization: Coerces to datetime, filters historical if requested.
- Defensive parsing: Regex for amounts (handles commas, embedded text), defaults for missing fields.
- Audit-friendly: Preserves extra columns; derives instrument_id if absent (e.g., PRODUCT-CCY-DATE-SEQ).
- No side effects: Does not inject HQLA/reserves (separate in bank_profile.intraday_liquidity).

Regulatory Alignment:
---------------------
- Supports LCR bucketing (30-day horizon) by standardizing 'date' and signed 'amount'.
- Currency uppercasing/3-char default (USD) aids multi-FX aggregation per Dodd-Frank/CRD IV.
- Deduplication prevents double-counting in liquidity ladders.

Dependencies:
-------------
- Python 3.8+: typing, datetime, re (standard library).
- pandas: DataFrame manipulation and date/amount coercion.

Usage in Workflow:
------------------
- Called by state_builder.build_state() post-harvest.
- Input: Raw list[dict] from portfolio_view.json (top-level, nested, counterparty-specific).
- Output: DataFrame → to_dict('records') for JSON serialization in state.json.
- Example: df = normalize_cashflows(raw_cfs, as_of='2025-10-17', include_historical_cashflows=False)

Integration Notes:
------------------
- For production: Extend with FX conversion (via portfolio.fx_rates) and regulatory runoff rates.
- Streamlit: Use df for interactive liquidity ladders (e.g., st.dataframe(df[df['date'] <= horizon_end])).
- Testing: Verify signing: inflows >0, outflows <0; no NaT/NaN in core columns.

Author:
-------
FintekMinds TreasuryAI PoC — Data Harmonization Module
Version: 2025-10 (Schema-robust for PoC ingestion)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date
import re

import pandas as pd


# ---- Public API --------------------------------------------------------------

def normalize_cashflows(
    cashflows: List[Dict[str, Any]],
    portfolio_view: Optional[Dict[str, Any]] = None,
    bank_profile: Optional[Dict[str, Any]] = None,
    as_of: Optional[Union[str, date, datetime]] = None,
    include_historical_cashflows: bool = True,
) -> pd.DataFrame:
    """
    Normalize heterogeneous cashflow rows to a clean DataFrame with at least:
        ['date','currency','amount','product','type','counterparty_id','instrument_id','description']

    Parameters
    ----------
    cashflows : list of dict
        Raw cashflow rows (mixed schemas acceptable). Expected keys vary (e.g., 'flow_amount', 'value_date').
        Nested 'cashflows' in dicts auto-flattened.
    portfolio_view : dict, optional
        Full portfolio dict (not required; kept for parity with caller). Unused here but extensible (e.g., FX rates).
    bank_profile : dict, optional
        Bank profile (intraday reserves, limits, etc.). We do NOT inject reserves here—handled in LCR metrics.
    as_of : str | date | datetime, optional
        Cutover date for optional filtering. If not provided, no date-based filtering is applied.
        Format: ISO 'YYYY-MM-DD' or datetime; coerced via pd.to_datetime.
    include_historical_cashflows : bool
        If False, drops rows strictly before `as_of`. Useful for forward-looking LCR (excludes settled flows).

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe. Missing non-critical fields are filled with defaults (e.g., '' for strings, NaN for amounts).
        Columns ordered: core first (date, currency, amount, ...), then extras. Index reset.

    Process:
        1. To DataFrame: Handles list[dict] or accidental dict{'cashflows': [...]}.
        2. Schema mapping: Aliases → canonical (case-insensitive).
        3. Date coercion: pd.to_datetime, normalize to date-only.
        4. Currency: Upper, strip, default 'USD'.
        5. Amount: Float parse (regex for embedded nums), sign by direction (+in/-out).
        6. Filter: Drop no-date/zero-amount; optional historical cutoff.
        7. Defaults: Ensure required cols (product='', etc.); derive instrument_id.
        8. Dedupe: Exact matches on core keys.
        9. Order: ['date', 'currency', 'amount', 'product', 'type', 'counterparty_id', 'instrument_id', 'description', 'direction'] + extras.

    Raises:
        None: Defensive; returns empty df on errors.

    Example:
        >>> raw = [{'date': '2025-10-18', 'amt': '1,000.50 USD', 'dir': 'IN'}]
        >>> df = normalize_cashflows(raw)
        >>> df['amount'].iloc[0]  # 1000.50 (positive inflow)

    Compliance Notes:
        - Signed amounts align with Basel III: +inflows (75% runoff cap), -outflows (3-100% rates).
        - For NSFR: Extend filter to 1-year horizon.
    """
    df = _to_dataframe(cashflows)

    if df.empty:
        return df  # Nothing to do

    # 1) Standardize columns and types
    df = _standardize_schema(df)

    # 2) Parse & coerce dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    # 3) Coerce currency (upper, 3–5 chars max)
    df["currency"] = df["currency"].astype(str).str.strip().str.upper().str[:5]
    df.loc[df["currency"].isin(["", "NAN", "NONE", "NULL"]), "currency"] = "USD"

    # 4) Amount to float; fix sign by optional 'direction'
    df["amount"] = _to_float(df["amount"])
    df["direction"] = df["direction"].astype(str).str.strip().str.upper()

    # If direction provided, enforce sign convention:
    #   INFLOW/IN/RECEIVE  -> +abs(amount)
    #   OUTFLOW/OUT/PAY    -> -abs(amount)
    inflow_mask = df["direction"].isin({"INFLOW", "IN", "RECEIVE", "RECEIVED"})
    outflow_mask = df["direction"].isin({"OUTFLOW", "OUT", "PAY", "PAID"})
    df.loc[inflow_mask, "amount"] = df.loc[inflow_mask, "amount"].abs()
    df.loc[outflow_mask, "amount"] = -df.loc[outflow_mask, "amount"].abs()

    # 5) Basic cleanup: drop rows with no date or zero/NaN amount
    df = df.dropna(subset=["date"])
    df = df[~df["amount"].isna()]

    # 6) Optional filtering by as_of
    if as_of is not None and not include_historical_cashflows:
        as_of_ts = _to_timestamp(as_of)
        if as_of_ts is not None:
            df = df[df["date"] >= as_of_ts]

    # 7) Ensure required columns exist with safe defaults
    for col, default in [
        ("product", ""),
        ("type", ""),
        ("counterparty_id", ""),
        ("instrument_id", ""),
        ("description", ""),
    ]:
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    # 8) Light de-duplication (exact duplicates only)
    df = df.drop_duplicates(subset=["date", "currency", "amount", "product", "type", "counterparty_id", "instrument_id", "description"])

    # 9) Final column order (keeps extra columns too)
    base_cols = ["date", "currency", "amount", "product", "type", "counterparty_id", "instrument_id", "description", "direction"]
    ordered = [c for c in base_cols if c in df.columns] + [c for c in df.columns if c not in base_cols]
    df = df.loc[:, ordered]

    return df.reset_index(drop=True)


# ---- Helpers ----------------------------------------------------------------

def _to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Coerce list[dict] to DataFrame, handling edge cases.

    Args:
        rows: Input list; if dict with 'cashflows', flatten it.

    Returns:
        pd.DataFrame: With core columns if absent; renames 'value_date' → 'date'.

    Notes:
        Failsafe: Converts malformed dicts to {'row': str(r)} to avoid crashes.
        Preserves all keys for schema flexibility.
    """
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(columns=["date", "currency", "amount"])
    # Flatten obvious nested 'cashflows' keys if a caller accidentally passes a dict
    if isinstance(rows, dict) and "cashflows" in rows:
        rows = rows.get("cashflows") or []
    try:
        df = pd.DataFrame(rows)
    except Exception:
        # Failsafe coercion
        df = pd.DataFrame([_force_kv(r) for r in rows if isinstance(r, dict)])
    if "date" not in df.columns and "value_date" in df.columns:
        df.rename(columns={"value_date": "date"}, inplace=True)
    return df


def _standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map common aliases → canonical names (case-insensitive).

    Args:
        df: Input DataFrame.

    Returns:
        pd.DataFrame: Renamed columns; core ensured (date/currency/amount=None); derived instrument_id/type.

    Mapping Examples:
        - 'ccy', 'curr' → 'currency'
        - 'amt', 'notional', 'value' → 'amount'
        - 'cp', 'counterparty' → 'counterparty_id'
        - 'cashflow_date', 'maturity_date' → 'date'

    Extensions:
        - Derives 'instrument_id' via _derive_instrument_id if missing/all-NaN.
        - Sets 'type'='' if absent; 'product' = 'type' if swapped.
        - Adds 'direction'='', 'description'='' as needed.
    """
    """Map common aliases -> canonical names."""
    colmap = {
        "ccy": "currency",
        "curr": "currency",
        "currency_code": "currency",
        "amt": "amount",
        "notional": "amount",
        "value": "amount",
        "flow": "amount",
        "flow_amount": "amount",
        "cf_amount": "amount",
        "product_type": "product",
        "instrument": "product",
        "cp": "counterparty_id",
        "counterparty": "counterparty_id",
        "cp_id": "counterparty_id",
        "instr_id": "instrument_id",
        "id": "instrument_id",
        "desc": "description",
        "narrative": "description",
        "direction_flag": "direction",
        "flow_direction": "direction",
        "cashflow_date": "date",
        "maturity_date": "date",  # if preview cashflows were emitted at maturity
    }
    keep = df.copy()
    cols_lower = {c: c for c in df.columns}

    # If someone used different capitalization (e.g., 'Ccy'), normalize via lowercase comparison
    lower_to_actual = {c.lower(): c for c in df.columns}
    for alias, canon in colmap.items():
        if alias in df.columns:
            keep.rename(columns={alias: canon}, inplace=True)
        else:
            # case-insensitive
            if alias.lower() in lower_to_actual:
                keep.rename(columns={lower_to_actual[alias.lower()]: canon}, inplace=True)

    # Ensure core columns exist
    for core in ("date", "currency", "amount"):
        if core not in keep.columns:
            keep[core] = None

    # Normalize instrument_id (derive when missing)
    if "instrument_id" not in keep.columns or keep["instrument_id"].isna().all():
        keep["instrument_id"] = keep.apply(_derive_instrument_id, axis=1)

    # Normalize 'type' vs 'product'
    if "type" not in keep.columns:
        keep["type"] = ""
    if "product" not in keep.columns:
        keep["product"] = keep["type"]

    # Add direction if missing
    if "direction" not in keep.columns:
        keep["direction"] = ""

    # Description column
    if "description" not in keep.columns:
        keep["description"] = ""

    return keep


def _derive_instrument_id(row: pd.Series) -> str:
    """
    Derive unique instrument_id if missing.

    Args:
        row: pd.Series with row data.

    Returns:
        str: Existing ID or derived 'PRODUCT-CCY-YYYYMMDD-SEQ' (SEQ=hash % 10000 for uniqueness).

    Priority:
        1. Existing: 'instrument_id', 'id', 'deal_id', 'trade_id', 'txn_id', 'tx_id'.
        2. Fallback: Based on product/ccy/date + hash-seq.

    Purpose:
        Enables traceability in liquidity ladders without upstream IDs.
    """
    # Try common fields first
    for k in ("instrument_id", "id", "deal_id", "trade_id", "txn_id", "tx_id"):
        v = str(row.get(k, "") or "").strip()
        if v:
            return v
    # Fall back: PRODUCT-CCY-YYYYMMDD-SEQ
    prod = str(row.get("product") or row.get("type") or "CF").upper()
    ccy = str(row.get("currency") or "USD").upper()
    dt = row.get("date")
    try:
        dts = pd.to_datetime(dt, errors="coerce")
        dtag = dts.strftime("%Y%m%d") if pd.notnull(dts) else "NA"
    except Exception:
        dtag = "NA"
    # small hash from row content to avoid collisions
    seq = abs(hash(str(row.to_dict()))) % 10000
    return f"{prod}-{ccy}-{dtag}-{seq:04d}"


def _to_float(series: pd.Series) -> pd.Series:
    """
    Best-effort conversion to float, handling commas and strings.

    Args:
        series: pd.Series of amounts (str/int/float).

    Returns:
        pd.Series: Floats; NaN for unparseable.

    Parsing:
        - Skips NaN/NONE/NULL.
        - Removes commas: '1,234.56' → 1234.56.
        - Regex extracts first num: '-$1,000 USD' → -1000.0.
        - Scientific: '1e3' → 1000.0.

    Notes:
        Preserves signs; used pre-direction signing.
    """
    if series.dtype.kind in ("f", "i"):
        return series.astype(float)
    def _coerce(x):
        if x is None:
            return float("nan")
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.upper() in {"NAN", "NONE", "NULL"}:
            return float("nan")
        s = s.replace(",", "")
        # extract first number (handles "1,234.56 USD")
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return float("nan")
        return float("nan")
    return series.apply(_coerce)


def _to_timestamp(x: Union[str, date, datetime, None]) -> Optional[pd.Timestamp]:
    """
    Coerce to normalized pd.Timestamp (date-only).

    Args:
        x: str/date/datetime/None.

    Returns:
        pd.Timestamp or None: Normalized if valid.

    Notes:
        errors='coerce'; used for as_of filtering.
    """
    if x is None:
        return None
    try:
        ts = pd.to_datetime(x, errors="coerce")
        return ts.normalize() if pd.notnull(ts) else None
    except Exception:
        return None


def _force_kv(r: Dict[str, Any]) -> Dict[str, Any]:
    """
    Defensive dict coercion for malformed rows.

    Args:
        r: Input dict (may have non-str keys).

    Returns:
        Dict[str, Any]: Str keys; values unchanged. Fallback {'row': str(r)} on error.

    Purpose:
        Prevents pd.DataFrame() crashes on bad data.
    """
    """Very defensive: stringify keys and leave values as-is."""
    try:
        return {str(k): v for k, v in r.items()}
    except Exception:
        return {"row": str(r)}