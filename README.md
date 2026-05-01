# Indian Mutual Fund Portfolio Analyzer

> **Live Demo:** [vedhmfproject.streamlit.app](https://vedhmfproject.streamlit.app/)

A full-stack data analytics web application that lets investors analyze, compare, and score any combination of Indian mutual funds using real AMFI NAV data — no synthetic numbers, no placeholders.

---

## What It Does

Pick any mutual funds from the AMFI universe and instantly get:

- Rolling returns across 1M, 3M, 6M, 1Y, and 3Y windows
- Annualized return, volatility, and Sharpe Ratio per fund
- Maximum drawdown with a visual drawdown chart
- Correlation matrix across all selected funds
- Benchmark comparison against Nifty 50 (via yfinance)
- Portfolio Health Score — a weighted composite out of 100
- AI Fund Recommender — 3 diversified portfolio options for your risk profile
- SIP Simulator — corpus projection and XIRR for any monthly SIP amount
- PDF Export — one-click downloadable report of the full analysis

---

## Portfolio Health Score

The health score is a weighted composite of four dimensions:

| Component         | Weight | What It Measures                        |
|-------------------|--------|-----------------------------------------|
| Sharpe Ratio      | 30%    | Risk-adjusted return quality            |
| Max Drawdown      | 25%    | Worst peak-to-trough loss               |
| Diversification   | 20%    | Average pairwise correlation (lower = better) |
| Consistent Returns| 25%    | Rolling return quality across windows   |

Scores above 70 are rated Excellent, 50-70 as Good, 30-50 as Fair, and below 30 as Needs Attention.

---

## AI Fund Recommender

The recommender engine scores funds from a curated universe of ~20 real Indian mutual funds across six categories: Large Cap, Flexi Cap, Mid Cap, Small Cap, Hybrid, ELSS, and Debt.

Each fund is scored on:
- 3Y and 1Y rolling returns (normalized)
- Annualized volatility (penalized)
- Sharpe Ratio

It then generates **3 distinct portfolio options** — Conservative, Balanced, and Growth — each containing 3 structurally diversified funds (no two funds from the same category). Each option shows the blended SIP projection across all 3 funds.

---

## Tech Stack

| Layer         | Library / Tool         |
|---------------|------------------------|
| Frontend      | Streamlit              |
| Data — NAV    | mftool (AMFI)          |
| Data — Index  | yfinance (Nifty 50)    |
| Calculations  | Pandas, NumPy          |
| Charts        | Plotly                 |
| PDF Export    | ReportLab              |
| Deployment    | Streamlit Community Cloud |

---

## Project Structure

```
Indian-Mutual-Fund-Portfolio-Analyzer/
|
|-- app.py                  # Main Streamlit entry point
|-- config.py               # Constants (rolling windows, risk-free rate, etc.)
|-- recommender.py          # AI fund recommendation engine
|-- report_generator.py     # PDF report builder (ReportLab)
|
|-- data/
|   |-- fetcher.py          # NAV fetching via mftool + yfinance benchmark
|   |-- cache_manager.py    # Local file-based cache for fund list
|
|-- calculations/
|   |-- returns.py          # Rolling, cumulative, annualized returns + SIP XIRR
|   |-- risk.py             # Volatility, Sharpe, drawdown, correlation
|   |-- scoring.py          # Portfolio Health Score logic
|
|-- ui/
|   |-- components.py       # All Streamlit rendering functions
|
|-- requirements.txt
|-- README.md
```

---

## Key Formulas

**Annualized Return**
```
(Final NAV / Initial NAV) ^ (252 / n_trading_days) - 1
```

**Sharpe Ratio** (risk-free rate = 6%, Indian T-bill proxy)
```
(Annualized Return - 0.06) / Annualized Volatility
```

**Max Drawdown**
```
min((NAV_t - Peak_t) / Peak_t)   for all t
```

**SIP Future Value** (used for projection)
```
FV = P * [((1 + r)^n - 1) / r] * (1 + r)
```

**SIP XIRR** — computed via Newton-Raphson on actual historical cashflows (not estimated).

---

## Data Sources

- **NAV History:** [AMFI India](https://www.amfiindia.com/) via the `mftool` Python library — covers all registered Indian mutual funds, updated daily
- **Benchmark:** Nifty 50 index via `yfinance` (ticker: `^NSEI`)
- **No synthetic data** — if a source is unavailable, the app shows a clear error message

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/Indian-Mutual-Fund-Portfolio-Analyzer.git
cd Indian-Mutual-Fund-Portfolio-Analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Python 3.10+ recommended. No API keys required — all data sources are free and public.

---

## Deployment

This app is deployed on **Streamlit Community Cloud** — free hosting for public Streamlit apps. The live version auto-deploys from the main branch on every push.

Live URL: [https://vedhmfproject.streamlit.app/](https://vedhmfproject.streamlit.app/)

---

## Disclaimer

This tool is for **informational and educational purposes only**. It does not constitute financial advice. Mutual fund investments are subject to market risks. Always read the scheme information document carefully before investing.

---

*Built with real data*
