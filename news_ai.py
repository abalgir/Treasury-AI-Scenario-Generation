from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, List
from langchain_core.messages import SystemMessage, HumanMessage
from with_whom.llm_factory import get_llm
from with_whom.infra.json_utils import parse_strict_json
from datetime import datetime, timedelta
from pathlib import Path
import requests
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

print("[news_ai] loaded v2025-10-15G (with recency & treasury focus)")

FMP_API_KEY = os.getenv("FMP_API_KEY")

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["as_of", "news_items", "explanation"],
    "properties": {
        "as_of": {"type": "string"},
        "news_items": {
            "type": "array",
            "maxItems": 10,
            "minItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "summary": {"type": "string"},
                    "pubDate": {"type": "string"},
                    "source": {"type": "string"},
                    "impact_level": {"type": "string", "enum": ["medium", "high"]},
                    "affected_markets": {"type": "array", "items": {"type": "string"}},
                    "liquidity_shock_type": {"type": "string",
                                             "enum": ["outflow_amplification", "inflow_haircut", "hqla_devaluation",
                                                      "funding_stress"]}  # New: Ties to Basel LCR/NSFR
                },
                "required": ["title", "url", "summary", "pubDate", "source", "impact_level", "affected_markets",
                             "liquidity_shock_type"]
            }
        },
        "explanation": {"type": "string"}
    }
}


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return json.dumps({"_unserializable": str(obj)}, ensure_ascii=False)


def _has_any_news(data: Dict[str, Any]) -> bool:
    return len(data.get("news_items", [])) > 0


def fetch_fmp_news(limit: int = 30) -> List[Dict[str, Any]]:
    if not FMP_API_KEY:
        raise ValueError("FMP_API_KEY not set in .env")
    url=(f"https://financialmodelingprep.com/stable/news/general-latest?page=0&limit=20&apikey={FMP_API_KEY}")
    # url = f"https://financialmodelingprep.com/api/v4/news/general?page=0&limit={limit}&apikey={FMP_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # New: Recency filter - last 7 days only (ISO pubDate check)
        cutoff = datetime.now() - timedelta(days=7)
        recent_data = [item for item in data if 'publishedDate' in item and datetime.fromisoformat(
            item['publishedDate'].replace('Z', '+00:00')[:19]) >= cutoff]
        print(f"[news_ai] Fetched {len(data)} total; filtered to {len(recent_data)} recent (last 7 days)")
        return recent_data if recent_data else data  # Fallback to all if none recent
    except Exception as e:
        raise RuntimeError(f"Failed to fetch FMP news: {e}")


def get_mock_news() -> List[Dict[str, Any]]:
    """Fallback mock for demo (3 synthetic recent items; remove in prod)."""
    current = datetime.now().isoformat()[:10]
    return [
        {
            "symbol": "",
            "title": "ECB Signals Rate Hold Amid Sticky Inflation",
            "url": "https://ecb.europa.eu/press/pr/date/2025/10/15",
            "text": "ECB maintains rates at 3.5% due to 2.5% HICP; potential wholesale funding stress for EU banks.",
            "publishedDate": f"{current}T10:00:00Z",
            "image": "",
            "topics": ["policy", "rates"],
            "uuid": "mock-ecb-1"
        },
        {
            "symbol": "",
            "title": "Fed Eyes 25bp Cut in December on Soft Data",
            "url": "https://federalreserve.gov/newsevents/pressreleases/2025/10/14",
            "text": "FOMC minutes hint at easing; FX volatility for USD/EUR pairs, impacting deposit runoffs.",
            "publishedDate": f"{current}T14:00:00Z",
            "image": "",
            "topics": ["policy", "fx"],
            "uuid": "mock-fed-1"
        },
        {
            "symbol": "",
            "title": "Geopolitical Tensions Escalate in Middle East",
            "url": "https://reuters.com/article/2025/10/15/geopolitics",
            "text": "Oil spike risks repo squeezes and HQLA devaluation in energy-exposed portfolios.",
            "publishedDate": f"{current}T16:00:00Z",
            "image": "",
            "topics": ["geopolitics", "commodities"],
            "uuid": "mock-geo-1"
        }
    ]


def _build_prompt(raw_news: List[Dict[str, Any]], retry: int = 0) -> str:
    current_date = datetime.now().strftime("%Y-%m-%d")
    sev_bump = f" (retry {retry + 1}: refine ranking for higher impact)" if retry > 0 else ""
    news_json = _safe_json(raw_news)

    return f"""
You are a financial market analysis expert specializing in treasury liquidity risks. From this exact list of recent news events (filtered to last 7 days; no external data, no hallucinations):
{news_json}

Rank them by potential impact on US/EU bank treasury (focus: liquidity shocks for USD/EUR/GBP portfolios under Basel III LCR/NSFR). Prioritize: policy shifts (e.g., rate holds triggering deposit runoff), geopolitics (funding stress), trade wars (credit spreads widening), market crashes (HQLA eligibility drops) over general sentiment. Ensure diversity: select top 5-10 from distinct topics (e.g., max 1-2 on rates; vary across macro/policy, FX, credit/liquidity).

For each selected: title (from data), url, summary (expand to max 5 lines on treasury implications like outflow amplification or inflow haircuts), pubDate (ISO format), source (publisher from data), impact_level (medium/high based on vol in rates/FX/credit/liquidity), affected_markets (e.g., ['US', 'EU']), liquidity_shock_type (outflow_amplification/inflow_haircut/hqla_devaluation/funding_stress).

Rules:
- Use only provided data; do not invent/add details or dates.
- Rank objectively: high = direct Basel trigger (e.g., retail panic > earnings miss). Enforce topic diversity.
- Output strictly conforms to JSON schema:
{_safe_json(SCHEMA)}

Example (use real from data):
{{
  "as_of": "{current_date}",
  "news_items": [
    {{
      "title": "Sample Title",
      "url": "https://example.com",
      "summary": "Treasury impl 1: Potential deposit runoff in EU.\\nImpl 2: FX vol on USD/EUR.\\nImpl 3: HQLA haircut risk.\\nImpl 4: Funding stress via repo.\\nImpl 5: Basel LCR impact.",
      "pubDate": "2025-10-15T10:00:00Z",
      "source": "Sample Publisher",
      "impact_level": "high",
      "affected_markets": ["US", "EU"],
      "liquidity_shock_type": "outflow_amplification"
    }}
  ],
  "explanation": "Ranking based on liquidity shock potential for treasury portfolios."
}}
Be precise, grounded in Basel III behavioral assumptions.{sev_bump}
"""


def generate_ai_news(max_retries: int = 3) -> Dict[str, Any]:
    llm = get_llm()
    try:
        raw_news = fetch_fmp_news(30)
    except Exception as e:
        print(f"[news_ai] FMP fetch failed: {e}; using mock fallback")
        raw_news = get_mock_news()

    if not raw_news:
        raise RuntimeError("No raw news fetched (even fallback).")

    data = None
    for retry in range(max_retries):
        system = SystemMessage(
            content="Respond with JSON ONLY matching schema. Ground in provided data only; focus on treasury liquidity shocks.")
        human = HumanMessage(content=_build_prompt(raw_news, retry))
        try:
            resp = llm.invoke([system, human])
            data = parse_strict_json(resp.content)
            print(
                "[news_ai] candidate_ok",
                "keys=", list(data.keys()),
                "news_count=", len(data.get("news_items", [])),
                "has_any_news=", _has_any_news(data)
            )
            if not _has_any_news(data):
                raise RuntimeError("Empty news after selection.")
        except Exception as e:
            print(f"[news_ai] Retry {retry + 1} failed: {e}")
            data = None

        if data and _has_any_news(data):
            break
        data = None

    if data is None:
        raise RuntimeError("Failed to generate valid news after retries.")

    data.setdefault("news_id", "AI_NEWS_1")
    return data


def save_to_data(data: Dict[str, Any], filename: str = "ai_news.json") -> Path:
    DATA_DIR = Path(__file__).resolve().parent / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"[news_ai] Saved to {filepath}")
    return filepath

def main():
    try:
        news = generate_ai_news()
        print(json.dumps(news, indent=4, ensure_ascii=False))

        payload = {
            "timestamp": datetime.now().isoformat(),
            "source": "news_ai",
            "news": news
        }
        save_to_data(payload)
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}, indent=2))

if __name__ == "__main__":
   main()