#!/usr/bin/env python3
# treasury_dashboard.py (updated)
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Treasury PoC Dashboard", layout="wide", initial_sidebar_state="collapsed")

DATA_DIR = Path(__file__).resolve().parent / "data"

@st.cache_data
def load_json_best_effort(name: str) -> Any:
    candidate_paths = [
        DATA_DIR / name,
        Path(__file__).resolve().parent / "data" / name,
        Path.cwd() / "data" / name,
        Path.cwd() / name,
    ]
    for p in candidate_paths:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                st.warning(f"Found {p} but failed to parse JSON: {e}")
                return {}
    st.info(f"{name} not found. Checked: {[str(p) for p in candidate_paths]}")
    return {}

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def fmt_usd(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "N/A"

def fmt_signed_usd(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    try:
        v = float(x)
        return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"
    except Exception:
        return "N/A"

def fmt_bn(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    try:
        xv = float(x)
        sign = "-" if xv < 0 else ""
        ax = abs(xv)
        if ax >= 1e9:
            return f"{sign}${ax / 1e9:,.3f}bn"
        if ax >= 1e6:
            return f"{sign}${ax / 1e6:,.0f}m"
        return f"{sign}${ax:,.0f}"
    except Exception:
        return "N/A"

def fmt_num(x: Optional[float], prec: int = 3) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.{prec}f}"
    except Exception:
        return "N/A"

def escape_dollars(s: str) -> str:
    return s.replace("$", r"\$") if isinstance(s, str) else s

scenario_results = load_json_best_effort("scenario_results.json") or {}
bank_profile = load_json_best_effort("bank_profile.json") or {}
counterparty_data = load_json_best_effort("counterparty_data.json") or []
portfolio_view = load_json_best_effort("portfolio_view.json") or {}
exchange_rate = load_json_best_effort("exchange_rate.json") or {}
ai_news = load_json_best_effort("ai_news.json") or {}
macro_data = load_json_best_effort("macro_data.json") or {}
baseline_metrics = load_json_best_effort("baseline_metrics.json") or {}

FX_MAP: Dict[str, float] = {}
if isinstance(exchange_rate, dict):
    for k, v in exchange_rate.items():
        try:
            FX_MAP[str(k).upper()] = float(v)
        except Exception:
            pass
if not FX_MAP and isinstance(portfolio_view, dict):
    fv = portfolio_view.get("fx_rates") or {}
    if isinstance(fv, dict):
        for k, v in fv.items():
            try:
                FX_MAP[str(k).upper()] = float(v)
            except Exception:
                pass
FX_MAP.setdefault("USD", 1.0)

def resolve_as_of() -> datetime.date:
    for src in (scenario_results, bank_profile, portfolio_view):
        if isinstance(src, dict):
            ao = (
                    src.get("as_of")
                    or src.get("reporting_snapshot", {}).get("as_of")
                    or src.get("baseline", {}).get("as_of")
            )
            if ao:
                try:
                    return datetime.fromisoformat(str(ao)[:10]).date()
                except Exception:
                    pass
    return datetime.utcnow().date()

AS_OF = resolve_as_of()

def bucket_for_days(days: int) -> str:
    if days <= 1: return "O/N"
    if days <= 7: return "1W"
    if days <= 14: return "2W"
    if days <= 30: return "1M"
    if days <= 90: return "3M"
    if days <= 180: return "6M"
    if days <= 365: return "1Y"
    return ">1Y"

def to_usd(amount: float, ccy: str, fx_map: Dict[str, float]) -> float:
    if amount is None:
        return 0.0
    try:
        rate = float(fx_map.get(str(ccy).upper(), 1.0))
    except Exception:
        rate = 1.0
    return float(amount) * rate

def flatten_counterparty_cashflows(cp_json: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for cp in (cp_json or []):
        cp_id = cp.get("counterparty_id") or cp.get("name") or ""
        for cf in (cp.get("cashflows") or []):
            dt_raw = cf.get("date") or cf.get("value_date")
            if not dt_raw:
                continue
            try:
                dt = datetime.fromisoformat(str(dt_raw).split("T")[0]).date()
            except Exception:
                continue
            rows.append({
                "counterparty": cp_id,
                "instrument_id": cf.get("instrument_id"),
                "type": (cf.get("type") or "").strip(),
                "product": (cf.get("product") or "").strip(),
                "currency": (cf.get("currency") or cf.get("ccy") or "USD"),
                "date": dt,
                "amount": safe_float(cf.get("amount"), 0.0),
                "direction": str(cf.get("direction") or "").strip().lower(),
            })
    if not rows:
        return pd.DataFrame(
            columns=["counterparty", "instrument_id", "type", "product", "currency", "date", "amount", "direction"])
    df = pd.DataFrame(rows)
    df["type_norm"] = df["type"].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    df["product_norm"] = df["product"].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    df["currency"] = df["currency"].astype(str).str.upper()
    return df

def build_ladder_dataframe_from_cp_aggregated(cp_df: pd.DataFrame, fx_map: Dict[str, float],
                                              as_of: datetime.date) -> pd.DataFrame:
    rows = []
    out_set = {"out", "pay", "debit", "payout", "-1", "-"}
    for _, r in cp_df.iterrows():
        d = r["date"]
        days = (d - as_of).days
        if days < 0:
            continue
        bucket = bucket_for_days(days)
        amt = safe_float(r["amount"])
        usd = to_usd(amt, r["currency"], fx_map)
        is_out = (r["direction"] in out_set) or (usd < 0)
        signed = -abs(usd) if is_out else abs(usd)
        product = r.get("product") or r.get("type") or "Unknown"
        rows.append({"product": product, "bucket": bucket, "amount_usd_signed": signed})
    if not rows:
        buckets = ["O/N", "1W", "2W", "1M", "3M", "6M", "1Y", ">1Y"]
        return pd.DataFrame(columns=["product"] + buckets).set_index("product")

    df = pd.DataFrame(rows)
    agg = df.groupby(["product", "bucket"])["amount_usd_signed"].sum().reset_index()

    buckets = ["O/N", "1W", "2W", "1M", "3M", "6M", "1Y", ">1Y"]
    wide = agg.pivot_table(index="product", columns="bucket", values="amount_usd_signed", aggfunc="sum").reindex(
        columns=buckets).fillna(0.0)
    wide = wide.sort_index()
    wide.loc["Total"] = wide.sum(axis=0)
    wide = wide.round(0).astype(float)
    return wide

def get_news_items(ai_news_obj: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], int]]:
    items = []
    idx_map: Dict[Tuple[str, str], int] = {}
    if isinstance(ai_news_obj, dict):
        items = (
                ai_news_obj.get("news", {}).get("news_items")
                or ai_news_obj.get("news_items")
                or []
        )
    for i, it in enumerate(items):
        key = (str(it.get("title")), str(it.get("url")))
        idx_map[key] = i
    return items, idx_map

ALL_NEWS, NEWS_INDEX = get_news_items(ai_news)

def numbered_news_label(item: Dict[str, Any]) -> str:
    key = (str(item.get("title")), str(item.get("url")))
    idx = NEWS_INDEX.get(key)
    idx_label = f"[{idx}]" if idx is not None else "[?]"
    src = item.get("source") or "N/A"
    dt = item.get("pubDate") or item.get("date") or "N/A"
    ttl = item.get("title") or "Untitled"
    return f"{idx_label} {ttl} — {src} ({dt})"

def render_news_analysis_expander():
    st.header("4. News Feed Analysis — AI Signal Sources")
    st.markdown(
        "_How AI ingests & prioritizes signals: High-impact items flagged for liquidity shocks (e.g., outflows, HQLA risks); summaries tie to Basel III behaviors._")
    if not ALL_NEWS:
        st.info("No news items loaded from ai_news.json.")
        return
    with st.expander("View Analyzed News Items (Ranked by Impact)", expanded=False):
        for item in ALL_NEWS:
            title = item.get("title", "Untitled")
            st.markdown(f"### {title}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Source", item.get("source", "N/A"))
            col2.metric("Date", item.get("pubDate", "N/A")[:10] if item.get("pubDate") else "N/A")
            col3.metric("Impact Level", item.get("impact_level", "N/A").upper())
            col4.metric("Markets Affected", ", ".join(item.get("affected_markets", [])))
            summary = item.get("summary", "")
            if summary:
                formatted_summary = summary.replace("\n", "  \n> ")
                st.markdown("**Analysis Summary:**")
                st.markdown(f"> {formatted_summary}")
            shock_type = item.get("liquidity_shock_type", "N/A")
            if shock_type != "N/A":
                st.caption(
                    f"_Shock Type: {shock_type.upper()} — e.g., amplifies LCR/NSFR under delayed policy signals._")
            st.divider()

st.title("Treasury PoC Dashboard — Treasurer View")
st.markdown("_Auditable AI scenarios for liquidity & market stress — treasurer-ready summaries_")

st.sidebar.header("Assumptions")
assumptions = scenario_results.get("assumptions", {}) or {}
st.sidebar.write(f"Inflow Cap (Stress): {int(assumptions.get('inflow_cap_stress', 0.75) * 100)}%")
basis_default = (assumptions.get("comparison_basis") or "gross")
basis_choice = st.sidebar.radio("Comparable LCR basis (for display only)", options=["gross", "net"],
                                index=(0 if basis_default == "gross" else 1))
st.sidebar.caption("Used to label comparable LCR deltas; values are from scenario_results.")

st.header("1. Baseline Liquidity Ladder ")
st.markdown(f"As-of: **{AS_OF.isoformat()}** — 30-day ladder (O/N → 1Y buckets); inflows positive, outflows negative")

cp_df = flatten_counterparty_cashflows(counterparty_data)
ladder_df = build_ladder_dataframe_from_cp_aggregated(cp_df, FX_MAP, AS_OF)

if ladder_df.empty:
    st.info("No cashflow rows found in counterparty_data.json.")
else:
    display = ladder_df.copy()
    for col in display.columns:
        display[col] = display[col].apply(fmt_signed_usd)
    st.dataframe(display, width="stretch", height=420)

st.header("2. Baseline Liquidity & Risk Indicators")
baseline = scenario_results.get("baseline", {}) or {}

c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
c1.markdown("**LCR (Effective)**")
c1.metric("LCR", fmt_num(baseline.get("lcr_effective"), 3), label_visibility="hidden")
c2.markdown("**Survival Days**")
c2.metric("Survival", fmt_num(baseline.get("survival_days_effective"), 1), label_visibility="hidden")
c3.markdown("**VaR 1d 99% (USD)**")
c3.metric("VaR", fmt_usd(baseline.get("var_1d_99_usd")), label_visibility="hidden")
c4.markdown("**HQLA Total (USD)**")
c4.metric("HQLA", fmt_usd(baseline.get("hqlatotal_usd")), label_visibility="hidden")

st.header("3. AI-Generated Scenarios — shocks, justification, behaviors & stressed results")

scenarios: List[Dict[str, Any]] = scenario_results.get("scenarios", []) or []
worst_id = (scenario_results.get("worst_case") or {}).get("scenario_id")

# If we have a worst-case id, put that scenario first
if worst_id and isinstance(scenarios, list):
    ordered = []
    worst_list = [s for s in scenarios if (s.get("id") or "") == worst_id]
    remainder = [s for s in scenarios if (s.get("id") or "") != worst_id]
    ordered.extend(worst_list)
    ordered.extend(remainder)
    scenarios = ordered

# helpers to compute deltas locally when missing
def to_nullable_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def compute_delta(baseline_val: Any, stressed_val: Any) -> Optional[float]:
    b = to_nullable_float(baseline_val)
    s = to_nullable_float(stressed_val)
    if b is None or s is None:
        return None
    return s - b

if not scenarios:
    st.info("No scenarios found in scenario_results.json")
else:
    for i, scen in enumerate(scenarios):
        sid = scen.get("id") or f"scenario_{i + 1:03d}"
        worst_tag = " — WORST CASE" if (worst_id and sid == worst_id) else ""
        title = f"Scenario {i + 1}: {sid}{worst_tag}"
        with st.expander(title, expanded=False):
            desc = scen.get("description") or ""
            if desc:
                st.subheader(desc)
            expl = scen.get("explanation") or ""
            if expl:
                st.markdown("**Explanation**")
                st.write(expl)

            shocks = scen.get("shocks_applied") or scen.get("shocks") or {}
            st.markdown("**Shocks (parameters found)**")
            if shocks and isinstance(shocks, dict):
                long_rows = []
                order = [
                    "rates_bps",
                    "credit_spread_bps",
                    "fx_shift_pct",
                    "liquidity_haircut_mult",
                    "ccy_outflow_mult",
                    "ccy_inflow_mult",
                    "payment_delay_days",
                    "outflow_accelerate_days",
                ]
                for factor in order:
                    val = shocks.get(factor)
                    if val in (None, {}, []):
                        continue
                    if isinstance(val, dict):
                        for k, v in val.items():
                            long_rows.append({"factor": factor, "param": str(k), "value": str(v)})
                    else:
                        long_rows.append({"factor": factor, "param": "", "value": str(val)})
                if long_rows:
                    shock_df = pd.DataFrame(long_rows, columns=["factor", "param", "value"])
                    for col in shock_df.columns:
                        shock_df[col] = shock_df[col].astype(str)
                    st.dataframe(shock_df, width="stretch", height=200)
                else:
                    st.info("No explicit shock parameters found.")
            else:
                st.info("No explicit shock parameters found in this scenario.")

            just_map = scen.get("justification_map") or {}
            if just_map:
                st.markdown("**Justification map**")
                try:
                    jm_df = pd.DataFrame([just_map]).T.reset_index()
                    jm_df.columns = ["key", "value"]
                    jm_df["value"] = jm_df["value"].astype(str)
                    st.dataframe(jm_df, width="stretch", height=180)
                except Exception:
                    st.json(just_map)

            news_refs = scen.get("news_refs") or []
            if news_refs:
                st.markdown("**Referenced news (from AI news feed)**")
                for item in news_refs[:12]:
                    st.write(numbered_news_label(item))

            st.markdown("**Behavioral adjustments (applied)**")
            beh = scen.get("behaviors_applied") or []
            if isinstance(beh, dict):
                beh = [beh]
            if not beh:
                st.info("No behavioral adjustments in this scenario.")
            else:
                for b in beh:
                    sent = b.get("treasurer_sentence")
                    if isinstance(sent, str) and sent.strip():
                        st.markdown(escape_dollars(sent))
                    else:
                        act = (b.get("action") or "").replace("_", " ").title()
                        crit = b.get("criteria") or {}
                        ccy = (crit.get("currency") or "USD")
                        itype = (crit.get("instrument_type") or "All instruments")
                        st.write(f"{act} — {ccy} — {itype} (details not available)")

            st.markdown("**Stressed results (liquidity & VaR)**")
            stressed = scen.get("stressed_indicators", {}) or {}
            diffs = scen.get("differences", {}) or {}
            comp_basis = stressed.get("comparison_basis") or (assumptions.get("comparison_basis") or "gross")

            # mapping of display rows -> (baseline key, stressed key, optional diff key)
            metric_map = [
                ("HQLA Total (USD)", "hqlatotal_usd", "hqlatotal_usd"),
                ("Gross Outflows (30d, USD)", "gross_outflows_30d_usd", "gross_outflows_30d_usd"),
                ("Inflows (30d, USD)", "inflows_30d_usd", "inflows_30d_usd"),
                ("Net Outflows (30d, USD)", "net_outflows_30d_usd", "net_outflows_30d_usd"),
                ("LCR (Gross)", "lcr_gross", "lcr_gross"),
                ("LCR (Net)", "lcr_net", "lcr_net"),
                (f"LCR (Comparable basis: {comp_basis})", "lcr_effective", "lcr_effective_comparable"),
                ("VaR 1d 99% (USD)", "var_1d_99_usd", "var_1d_99_usd"),
            ]

            rows = []
            for label, bkey, skey in metric_map:
                base_val = baseline.get(bkey)
                stressed_val = stressed.get(skey)
                # try to get diff from scenario differences first
                diff_val = diffs.get(skey) if diffs and skey in diffs else None
                if diff_val is None:
                    # on-the-fly compute if both baseline and stressed numeric
                    computed = compute_delta(base_val, stressed_val)
                    diff_val = computed
                # format appropriately
                if bkey.endswith("_usd") or bkey in ("hqlatotal_usd", "var_1d_99_usd"):
                    b_display = fmt_usd(base_val)
                    s_display = fmt_usd(stressed_val)
                    d_display = fmt_usd(diff_val)
                else:
                    # LCR-like metric (decimal)
                    b_display = fmt_num(base_val, 3)
                    s_display = fmt_num(stressed_val, 3)
                    d_display = fmt_num(diff_val, 3)
                rows.append((label, b_display, s_display, d_display))

            comp_df = pd.DataFrame(rows, columns=["Metric", "Baseline", "Stressed", "Delta"])
            for col in ["Baseline", "Stressed", "Delta"]:
                comp_df[col] = comp_df[col].astype(str)
            st.dataframe(comp_df, width="stretch", height=300)

            sidecar = scen.get("audit_log_sidecar")
            if sidecar:
                st.caption(f"Audit log: {sidecar}")

render_news_analysis_expander()

st.markdown("---")
st.caption("*Designed for treasury review — auditable, transparent scenario summaries.*")
