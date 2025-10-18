# Treasury AI — Proof of Concept (PoC)

**Proof-of-Concept** showing how AI can generate liquidity stress scenarios and explain/justify them to a treasurer.
This project is experimental research code and **not** production-ready.

> ⚠️ **Important legal / safety notes**
>
> * This is a research Proof-of-Concept. Use at your own risk.
> * The code is provided *as-is*; maintainers and contributors are **not responsible** for decisions made using outputs of this software.
> * **Do not** run this on production systems or with sensitive/PII data unless you have fully audited every component.
> * **Data might differ from the white paper one.**

---

## Project layout (important files)

```
py_code/with_whom/
├─ cf_normalizer.py
├─ state_builder.py
├─ portfolio_aggregator.py
├─ counterpart_aggregator.py
├─ metrics.py
├─ macro_market_data.py
├─ news_ai.py
├─ macro_ai.py
├─ scenario_builder.py
├─ orchestrator_phase1.py   <- orchestrator that runs the sequence
├─ treasury_dashboard.py
│  
└─ data/
   ├─ portfolio_view.json
   ├─ counterparty_data.json
   ├─ bank_profile.json
   ├─ exchange_rate.json
   ├─ state.json
   ├─ scenario_results.json
   └─ (other generated artifacts)
```

---

## Main pipeline sequence

The orchestrator runs the components in the following order (this is the sequence you should expect the pipeline to execute):

1. `portfolio_aggregator`  → writes `portfolio_view.json`
2. `state_builder`         → writes `state.json`
3. `counterpart_aggregator`→ writes `portfolio_intel.json` (and/or `counterparty_data.json`)
4. `metrics`               → writes `baseline_metrics.json`
5. `macro_market_data`     → writes `macro_data.json`
6. `news_ai`               → writes `ai_news.json`
7. `macro_ai`              → writes `standalone_scenarios.json`
8. `scenario_builder`      → writes `scenario_results.json`

After this sequence is completed, run the Streamlit dashboard to visualize the results:

```bash
streamlit run py_code/with_whom/dashboard/treasury_dashboard.py
```

> Tip: `orchestrator_phase1.py` may simply call these modules in that sequence; run it from repo root to produce the JSON artifacts used by the dashboard.

---

## JSON files required by the dashboard & pipeline

Place these files in `py_code/with_whom/data/` (or the `data/` folder the scripts expect). Minimal skeletons are shown below.

### `counterparty_data.json`

A list of counterparties and their cashflows. Required by the dashboard ladder view.

```json
[
  {
    "counterparty_id": "CP-001",
    "name": "Example Bank",
    "cashflows": [
      {
        "date": "2025-10-20",
        "amount": 1000000,
        "currency": "USD",
        "direction": "IN",
        "instrument_id": "BOND-123",
        "product": "Bond",
        "type": "cash"
      }
    ]
  }
]
```

### `bank_profile.json`

Bank liquidity profile and HQLA breakdown. Used by `state_builder` and metrics.

```json
{
  "as_of": "2025-10-18",
  "liquidity": {
    "hqla_breakdown": {
      "Level 1": {"amount_usd": 2000000000, "currency": "USD", "haircut": 0.0},
      "Level 2A": {"amount_usd": 750000000, "currency": "USD", "haircut": 0.15}
    },
    "total_hqla_usd": 2750000000,
    "survival_days_override": null
  },
  "intraday_liquidity": {"reserve": 50000000, "currency": "USD"}
}
```

### `exchange_rate.json`

Simple currency → USD rate map:

```json
{
  "USD": 1.0,
  "EUR": 1.07,
  "GBP": 1.25
}
```

---

## How to run

### 1) Setup environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows PowerShell

# Install dependencies (example)
pip install -r requirements.txt
```

**Suggested `requirements.txt`**

```
pandas>=1.4
streamlit>=1.18
python-dateutil
```

### 2) Run orchestrator (produce JSON outputs)

From repo root:

```bash
python py_code/with_whom/orchestrator_phase1.py
# or if you use scenario_builder directly
python py_code/with_whom/scenario_builder.py
```

This will produce the JSON files in `py_code/with_whom/data/` used by the dashboard.

### 3) Run Streamlit dashboard

```bash
streamlit run py_code/with_whom/dashboard/treasury_dashboard.py
```

Open the browser tab Streamlit gives you (usually `http://localhost:8501`).

---


## Contributing

Fork the repository and open PRs. For changes affecting normalization or scenario logic, include:

* A small sanitized input sample
* Expected output JSON for regression testing
* Unit tests if possible

---

## License & disclaimer

This PoC is provided *as-is* for research/education. Use at your own risk. The maintainers/contributors are not responsible for decisions made using outputs from this code.
**Data might differ from the white paper one.**

MIT License

Copyright (c) 2025 KPIMinds LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
