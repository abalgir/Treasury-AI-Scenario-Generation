#!/usr/bin/env python3
# cf_normalizer.py
# Drop-in normalizer for state_builder.build_state(...)
# - Returns a pandas.DataFrame with normalized columns.
# - Ensures returned DataFrame has unique column names so df.to_dict(...) doesn't omit data.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date
import re
import json
import sys
import traceback
from pathlib import Path

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

    Returns a DataFrame whose columns are guaranteed unique (duplicates renamed with __dupN).
    """
    df = _to_dataframe(cashflows)

    if df.empty:
        return df  # Nothing to do

    # 1) Standardize columns and types
    df = _standardize_schema(df)

    # Ensure unique column names (prevents pandas warnings and df.to_dict omissions)
    df = _ensure_unique_columns(df)

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

    # 7) Ensure required columns exist with safe defaults (after unique-ification)
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
    drop_subset = [c for c in ["date", "currency", "amount", "product", "type", "counterparty_id", "instrument_id", "description"] if c in df.columns]
    if drop_subset:
        df = df.drop_duplicates(subset=drop_subset)

    # 9) Final column order (keeps extra columns too)
    base_cols = ["date", "currency", "amount", "product", "type", "counterparty_id", "instrument_id", "description", "direction"]
    ordered = [c for c in base_cols if c in df.columns] + [c for c in df.columns if c not in base_cols]
    df = df.loc[:, ordered]

    # Ensure uniqueness again before returning
    df = _ensure_unique_columns(df)
    return df.reset_index(drop=True)


# ---- Helpers ----------------------------------------------------------------

def _to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Coerce list[dict] to DataFrame, handling edge cases.
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
    """
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
        "maturity_date": "date",
    }
    keep = df.copy()

    # case-insensitive mapping
    lower_to_actual = {c.lower(): c for c in df.columns}
    for alias, canon in colmap.items():
        if alias in df.columns:
            keep.rename(columns={alias: canon}, inplace=True)
        else:
            if alias.lower() in lower_to_actual:
                keep.rename(columns={lower_to_actual[alias.lower()]: canon}, inplace=True)

    # Ensure core columns exist
    for core in ("date", "currency", "amount"):
        if core not in keep.columns:
            keep[core] = None

    # Normalize instrument_id (derive when missing OR all-NA)
    need_instrument_id = False
    if "instrument_id" not in keep.columns:
        need_instrument_id = True
    else:
        col = keep["instrument_id"]
        try:
            is_all_na = bool(col.isna().all())
        except Exception:
            # defensive: if .isna().all() returns Series (rare), collapse
            try:
                is_all_na = bool(col.isna().all().all())
            except Exception:
                is_all_na = False
        if is_all_na:
            need_instrument_id = True

    if need_instrument_id:
        # apply row-by-row safe derivation for instrument ids
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
    """
    for k in ("instrument_id", "id", "deal_id", "trade_id", "txn_id", "tx_id"):
        try:
            v = str(row.get(k, "") or "").strip()
            if v:
                return v
        except Exception:
            continue
    prod = str(row.get("product") or row.get("type") or "CF").upper()
    ccy = str(row.get("currency") or "USD").upper()
    dt = row.get("date")
    try:
        dts = pd.to_datetime(dt, errors="coerce")
        dtag = dts.strftime("%Y%m%d") if pd.notnull(dts) else "NA"
    except Exception:
        dtag = "NA"
    seq = abs(hash(str(row.to_dict()))) % 10000
    return f"{prod}-{ccy}-{dtag}-{seq:04d}"


def _to_float(series: pd.Series) -> pd.Series:
    """
    Best-effort conversion to float, handling commas and strings.
    """
    if getattr(series, "dtype", None) is not None and series.dtype.kind in ("f", "i"):
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
    """
    try:
        return {str(k): v for k, v in r.items()}
    except Exception:
        return {"row": str(r)}


def _ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame columns are unique. If duplicates exist, rename subsequent
    occurrences to <col>__dupN where N starts at 1.

    This preserves all columns and avoids pandas warnings and data omission
    on df.to_dict(orient='records').
    """
    cols = list(df.columns)
    seen = {}
    new_cols: List[str] = []
    for col in cols:
        if col not in seen:
            seen[col] = 1
            new_cols.append(col)
        else:
            seen[col] += 1
            # create a unique suffix
            suffix = f"__dup{seen[col]-1}"
            candidate = f"{col}{suffix}"
            # avoid accidental collision
            while candidate in seen:
                seen[col] += 1
                suffix = f"__dup{seen[col]-1}"
                candidate = f"{col}{suffix}"
            seen[candidate] = 1
            new_cols.append(candidate)
    # If nothing changed, return original df
    if new_cols == cols:
        return df
    df = df.copy()
    df.columns = new_cols
    return df


# ---------------------------
# Small CLI / debug entrypoint to reproduce errors directly
# ---------------------------
def main():
    """
    1) Attempts to read canonical data files from ../data (portfolio_view.json or counterparty_data.json).
    2) Calls normalize_cashflows on the discovered cashflows and prints DataFrame info.
    """
    BASE = Path(__file__).resolve().parent
    DATA_DIR = BASE / "data"
    print(f"[INFO] cf_normalizer main running. data dir: {DATA_DIR}")

    pv_path = DATA_DIR / "portfolio_view.json"
    cp_path = DATA_DIR / "counterparty_data.json"
    state_path = DATA_DIR / "state.json"

    cashflows = None
    as_of = None

    try:
        if pv_path.exists():
            print(f"[INFO] Loading {pv_path}")
            pv = json.loads(pv_path.read_text())
            if isinstance(pv, dict) and pv.get("cashflows"):
                cashflows = pv.get("cashflows")
            elif isinstance(pv, dict) and pv.get("counterparties"):
                cplist = pv.get("counterparties") or []
                agg = []
                for cp in cplist:
                    if isinstance(cp, dict):
                        cfs = cp.get("cashflows") or cp.get("cash_flows") or []
                        if isinstance(cfs, list):
                            agg.extend(cfs)
                cashflows = agg
            if isinstance(pv, dict) and pv.get("as_of"):
                as_of = pv.get("as_of")
        elif cp_path.exists():
            print(f"[INFO] Loading {cp_path}")
            cps = json.loads(cp_path.read_text())
            agg = []
            if isinstance(cps, list):
                for cp in cps:
                    if isinstance(cp, dict):
                        cfs = cp.get("cashflows") or cp.get("cash_flows") or []
                        if isinstance(cfs, list):
                            agg.extend(cfs)
            cashflows = agg
        elif state_path.exists():
            print(f"[INFO] Loading {state_path}")
            st = json.loads(state_path.read_text())
            if isinstance(st, dict) and st.get("cashflows"):
                cashflows = st.get("cashflows")
            elif isinstance(st, dict) and st.get("state") and isinstance(st.get("state"), dict):
                inner = st.get("state")
                if inner.get("cashflows"):
                    cashflows = inner.get("cashflows")
            if isinstance(st, dict) and (st.get("as_of") or (st.get("state") or {}).get("as_of")):
                as_of = st.get("as_of") or (st.get("state") or {}).get("as_of")
        else:
            print("[WARN] No portfolio_view.json / counterparty_data.json / state.json found in data dir. Exiting.")
            return

        if not isinstance(cashflows, list) or not cashflows:
            print("[WARN] No cashflows discovered (empty or not a list). Exiting.")
            return

        print(f"[INFO] Calling normalize_cashflows on {len(cashflows)} raw rows (as_of={as_of})")
        df = normalize_cashflows(cashflows, portfolio_view=None, bank_profile=None, as_of=as_of, include_historical_cashflows=False)
        print("[OK] normalize_cashflows returned DataFrame:")
        try:
            print(df.info())
            print(df.head(10).to_string(index=False))
        except Exception as e:
            print("[WARN] exception while printing DataFrame info:", e)
            traceback.print_exc()

    except Exception as e:
        print("[ERROR] normalize_cashflows raised an exception:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run cf_normalizer.normalize_cashflows on local data files for debugging.")
    parser.add_argument("--data-dir", "-d", default=os.path.join(os.path.dirname(__file__), "data"),
                        help="Directory containing portfolio_view.json / cashflow files (default: ./data next to this script)")
    parser.add_argument("--print-sample", action="store_true", help="Print sample rows from normalized DataFrame")
    parser.add_argument("--include-historical", action="store_true", help="Pass include_historical_cashflows=True to normalize (default False)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"[ERROR] data directory does not exist: {data_dir}")
        raise SystemExit(1)

    # Find candidate JSONs and aggregate
    candidates = []
    for name in ("portfolio_view.json", "state.json", "portfolio_view_*.json"):
        candidates.extend(sorted(data_dir.glob(name)))
    candidates.extend(sorted(p for p in data_dir.glob("*cashflow*.json")))
    candidates.extend(sorted(data_dir.glob("*.json")))

    seen = set()
    files = []
    for p in candidates:
        if str(p) not in seen:
            files.append(p)
            seen.add(str(p))

    if not files:
        print(f"[WARN] No JSON files found in {data_dir}")
        raise SystemExit(0)

    all_cashflows = []
    portfolio_as_of = None

    def try_load_json(path: Path):
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
            return None

    for fpath in files:
        print(f"[INFO] Loading {fpath}")
        obj = try_load_json(fpath)
        if obj is None:
            continue

        if isinstance(obj, dict) and "as_of" in obj and not portfolio_as_of:
            portfolio_as_of = obj.get("as_of")

        if isinstance(obj, list):
            print(f"[INFO] Found top-level list in {fpath} (assuming cashflows, len={len(obj)})")
            all_cashflows.extend(obj)
            continue

        cf = None
        if isinstance(obj, dict):
            if "cashflows" in obj and isinstance(obj["cashflows"], list):
                cf = obj["cashflows"]
                print(f"[INFO] Found 'cashflows' in {fpath} (len={len(cf)})")
            elif "cashflows_preview" in obj:
                cp = obj["cashflows_preview"]
                if isinstance(cp, dict):
                    if "baseline" in cp and isinstance(cp["baseline"], list):
                        cf = cp["baseline"]
                        print(f"[INFO] Found 'cashflows_preview'->'baseline' in {fpath} (len={len(cf)})")
                    else:
                        flattened = []
                        for v in cp.values():
                            if isinstance(v, list):
                                flattened.extend(v)
                        if flattened:
                            cf = flattened
                            print(f"[INFO] Flattened cashflows_preview lists in {fpath} (len={len(cf)})")
                elif isinstance(cp, list):
                    cf = cp
                    print(f"[INFO] Found 'cashflows_preview' as list in {fpath} (len={len(cf)})")
            elif "flows" in obj and isinstance(obj["flows"], list):
                cf = obj["flows"]
                print(f"[INFO] Found 'flows' in {fpath} (len={len(cf)})")
            elif "rows" in obj and isinstance(obj["rows"], list):
                cf = obj["rows"]
                print(f"[INFO] Found 'rows' in {fpath} (len={len(cf)})")
            elif "data" in obj and isinstance(obj["data"], list):
                cf = obj["data"]
                print(f"[INFO] Found 'data' in {fpath} (len={len(cf)})")

        if cf:
            all_cashflows.extend(cf)

    print(f"[INFO] Total raw cashflow rows collected: {len(all_cashflows)}")

    if not all_cashflows:
        print("[WARN] No cashflows discovered (empty or not a list). Exiting.")
        raise SystemExit(0)

    try:
        df = normalize_cashflows(all_cashflows, portfolio_view=None, bank_profile=None,
                                 as_of=portfolio_as_of, include_historical_cashflows=bool(args.include_historical))

        print(f"[INFO] normalize_cashflows returned DataFrame with shape: {getattr(df, 'shape', None)}")
        try:
            print("\n[INFO] DataFrame.dtypes:")
            print(df.dtypes)
            print("\n[INFO] DataFrame.info():")
            df.info()
        except Exception:
            pass

        if args.print_sample:
            print("\n[INFO] Sample rows (up to 10):")
            print(df.head(10).to_dict(orient="records"))

    except Exception as exc:
        print("[ERROR] Exception raised while normalizing cashflows:")
        traceback.print_exc()
        raise
