"""
recommender.py
--------------
Rule-based mutual fund recommendation engine — 3 Model Portfolios.

Flow:
  1. Curated fund universe of real Indian MFs with category tags.
  2. User inputs: risk appetite, investment horizon, monthly SIP amount.
  3. Three pre-defined model portfolios per risk level, each with a fixed
     category mix — diversification is baked into the structure.
  4. For each portfolio, pick the best-scoring fund per category slot.
  5. Return 3 portfolio options, each with 3 funds, blended SIP projection,
     and plain-English reasoning.
"""

import numpy as np
import pandas as pd
from typing import Optional

from data.fetcher import get_nav_history
from calculations.returns import annualized_return, latest_rolling_returns
from calculations.risk import annualized_volatility, sharpe_ratio, max_drawdown


# ---------------------------------------------------------------------------
# Fund Universe
# ---------------------------------------------------------------------------

FUND_UNIVERSE = [
    # Large Cap
    ("120503", "Mirae Asset Large Cap Fund",       "Large Cap"),
    ("118989", "Axis Bluechip Fund",               "Large Cap"),
    ("112090", "ICICI Pru Bluechip Fund",          "Large Cap"),
    ("101206", "SBI Blue Chip Fund",               "Large Cap"),
    ("120716", "Canara Robeco Bluechip Equity",    "Large Cap"),
    # Flexi Cap
    ("118778", "Parag Parikh Flexi Cap Fund",      "Flexi Cap"),
    ("112102", "HDFC Flexi Cap Fund",              "Flexi Cap"),
    ("125354", "UTI Flexi Cap Fund",               "Flexi Cap"),
    # Mid Cap
    ("119598", "Kotak Emerging Equity Fund",       "Mid Cap"),
    ("120847", "DSP Midcap Fund",                  "Mid Cap"),
    ("118701", "HDFC Mid-Cap Opportunities Fund",  "Mid Cap"),
    ("100341", "Franklin India Prima Fund",        "Mid Cap"),
    # Small Cap
    ("120828", "Axis Small Cap Fund",              "Small Cap"),
    ("125497", "Nippon India Small Cap Fund",      "Small Cap"),
    ("148627", "SBI Small Cap Fund",               "Small Cap"),
    ("119775", "Kotak Small Cap Fund",             "Small Cap"),
    # Hybrid
    ("101343", "HDFC Balanced Advantage Fund",     "Hybrid"),
    ("119551", "Kotak Equity Hybrid Fund",         "Hybrid"),
    ("106655", "ICICI Pru Equity & Debt Fund",     "Hybrid"),
    # ELSS
    ("120716", "Canara Robeco Equity Tax Saver",   "ELSS"),
    ("112090", "ICICI Pru Long Term Equity Fund",  "ELSS"),
    ("101206", "SBI Long Term Equity Fund",        "ELSS"),
    # Debt
    ("119270", "HDFC Corporate Bond Fund",         "Debt"),
    ("118701", "Kotak Bond Short Term Fund",       "Debt"),
    ("120716", "SBI Magnum Medium Duration Fund",  "Debt"),
]

# Deduplicate by scheme_code
_seen_codes: set = set()
FUND_UNIVERSE_CLEAN = []
for _entry in FUND_UNIVERSE:
    if _entry[0] not in _seen_codes:
        _seen_codes.add(_entry[0])
        FUND_UNIVERSE_CLEAN.append(_entry)


# ---------------------------------------------------------------------------
# Model Portfolio Templates
# ---------------------------------------------------------------------------
# Three portfolios per risk level. Each portfolio defines 3 category slots.
# Best-scoring fund per slot is selected — guarantees category diversity.

MODEL_PORTFOLIOS = {
    "Low": [
        {
            "name":        "Option A — Capital Protection",
            "description": "Maximum safety. Focused on low-risk debt and balanced funds.",
            "emoji":       "🛡️",
            "slots":       ["Debt", "Debt", "Hybrid"],
        },
        {
            "name":        "Option B — Stable Income",
            "description": "Slight growth tilt with hybrid and one large-cap for mild upside.",
            "emoji":       "⚖️",
            "slots":       ["Debt", "Hybrid", "Large Cap"],
        },
        {
            "name":        "Option C — Conservative Growth",
            "description": "More equity exposure while staying in the low-risk zone.",
            "emoji":       "🌱",
            "slots":       ["Hybrid", "Large Cap", "Flexi Cap"],
        },
    ],
    "Medium": [
        {
            "name":        "Option A — Safety First",
            "description": "Growth with a safety net. Anchored by large caps and a hybrid.",
            "emoji":       "🛡️",
            "slots":       ["Large Cap", "Hybrid", "ELSS"],
        },
        {
            "name":        "Option B — Balanced Growth",
            "description": "Equal spread across large, flexi, and hybrid for steady compounding.",
            "emoji":       "⚖️",
            "slots":       ["Large Cap", "Flexi Cap", "Hybrid"],
        },
        {
            "name":        "Option C — Growth Oriented",
            "description": "Higher return potential with flexi-cap and ELSS for tax savings.",
            "emoji":       "🚀",
            "slots":       ["Flexi Cap", "Large Cap", "ELSS"],
        },
    ],
    "High": [
        {
            "name":        "Option A — Aggressive Growth",
            "description": "Broad equity exposure — large, flexi, and mid cap for high growth.",
            "emoji":       "🚀",
            "slots":       ["Large Cap", "Flexi Cap", "Mid Cap"],
        },
        {
            "name":        "Option B — High Conviction",
            "description": "Mid and small cap tilt for alpha seekers with a strong risk appetite.",
            "emoji":       "⚡",
            "slots":       ["Flexi Cap", "Mid Cap", "Small Cap"],
        },
        {
            "name":        "Option C — Maximum Alpha",
            "description": "All-in on high-growth equity — mid, small cap and ELSS for tax benefit.",
            "emoji":       "🔥",
            "slots":       ["Mid Cap", "Small Cap", "ELSS"],
        },
    ],
}


# ---------------------------------------------------------------------------
# Horizon Config
# ---------------------------------------------------------------------------

HORIZON_WEIGHTS = {
    "Short (1Y)":  {"sharpe": 0.25, "returns": 0.25, "drawdown": 0.30, "volatility": 0.20},
    "Medium (3Y)": {"sharpe": 0.40, "returns": 0.30, "drawdown": 0.20, "volatility": 0.10},
    "Long (5Y+)":  {"sharpe": 0.40, "returns": 0.40, "drawdown": 0.15, "volatility": 0.05},
}

HORIZON_RETURN_WINDOW = {
    "Short (1Y)":  "1Y",
    "Medium (3Y)": "3Y",
    "Long (5Y+)":  "5Y",
}


# ---------------------------------------------------------------------------
# Metric Fetcher
# ---------------------------------------------------------------------------

def _fetch_fund_metrics(scheme_code: str, horizon: str) -> Optional[dict]:
    try:
        nav_series = get_nav_history(scheme_code)
        if nav_series is None or len(nav_series) < 60:
            return None

        nav_df    = nav_series.to_frame(name=scheme_code)
        daily_ret = nav_df.pct_change().dropna()

        ann_ret  = annualized_return(nav_df)
        ann_vol  = annualized_volatility(daily_ret)
        sharpe_s = sharpe_ratio(daily_ret)
        mdd_s    = max_drawdown(nav_df)

        window_key = HORIZON_RETURN_WINDOW.get(horizon, "1Y")
        rolling_df = latest_rolling_returns(nav_df)

        if window_key in rolling_df.index:
            horizon_return = float(rolling_df.loc[window_key, scheme_code])
        elif "1Y" in rolling_df.index:
            horizon_return = float(rolling_df.loc["1Y", scheme_code])
        else:
            horizon_return = float(ann_ret.iloc[0]) if not ann_ret.empty else np.nan

        return {
            "scheme_code":    scheme_code,
            "ann_return":     float(ann_ret.iloc[0])  if not ann_ret.empty  else np.nan,
            "ann_volatility": float(ann_vol.iloc[0])  if not ann_vol.empty  else np.nan,
            "sharpe":         float(sharpe_s.iloc[0]) if not sharpe_s.empty else np.nan,
            "max_drawdown":   float(mdd_s.iloc[0])    if not mdd_s.empty    else np.nan,
            "horizon_return": horizon_return,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _normalize(value: float, low: float, high: float, invert: bool = False) -> float:
    if np.isnan(value) or high == low:
        return 50.0
    score = (value - low) / (high - low) * 100.0
    score = max(0.0, min(100.0, score))
    return (100.0 - score) if invert else score


def _score_fund(metrics: dict, all_metrics: list, weights: dict) -> float:
    if not all_metrics:
        return 0.0

    def _vals(key):
        return [m[key] for m in all_metrics if not np.isnan(m[key])]

    sharpe_vals = _vals("sharpe")
    ret_vals    = _vals("horizon_return")
    dd_vals     = _vals("max_drawdown")
    vol_vals    = _vals("ann_volatility")

    s_sharpe  = _normalize(metrics["sharpe"],         min(sharpe_vals, default=0), max(sharpe_vals, default=1))
    s_returns = _normalize(metrics["horizon_return"], min(ret_vals,    default=0), max(ret_vals,    default=1))
    s_dd      = _normalize(metrics["max_drawdown"],   min(dd_vals,     default=-1), max(dd_vals,    default=0), invert=True)
    s_vol     = _normalize(metrics["ann_volatility"], min(vol_vals,    default=0), max(vol_vals,    default=1), invert=True)

    return round(
        s_sharpe  * weights["sharpe"]   +
        s_returns * weights["returns"]  +
        s_dd      * weights["drawdown"] +
        s_vol     * weights["volatility"],
        1
    )


# ---------------------------------------------------------------------------
# Best Fund Picker per Slot
# ---------------------------------------------------------------------------

def _pick_best_for_slot(category: str, scored_metrics: list, already_used: set) -> Optional[dict]:
    candidates = [
        m for m in scored_metrics
        if m["category"] == category and m["scheme_code"] not in already_used
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda m: m["score"])


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------

def _fund_reason(metrics: dict, category: str, slot_index: int) -> str:
    ret    = metrics.get("horizon_return", np.nan)
    sharpe = metrics.get("sharpe", np.nan)
    mdd    = metrics.get("max_drawdown", np.nan)

    parts = []
    if not np.isnan(ret):
        parts.append(f"{ret*100:.1f}% horizon return")
    if not np.isnan(sharpe) and sharpe > 0:
        parts.append(f"Sharpe {sharpe:.2f}")
    if not np.isnan(mdd):
        parts.append(f"drawdown {mdd*100:.1f}%")

    metric_str = " · ".join(parts) if parts else "limited history available"
    return f"Best {category} pick — {metric_str}"


def _portfolio_summary(option: dict, risk: str, horizon: str) -> str:
    return (
        f"{option['description']} "
        f"Suited for {risk.lower()} risk investors with a {horizon.lower()} horizon."
    )


# ---------------------------------------------------------------------------
# SIP Helpers
# ---------------------------------------------------------------------------

def sip_projection(monthly_sip: float, annual_return: float, years: int) -> float:
    if np.isnan(annual_return) or annual_return <= 0:
        return monthly_sip * 12 * years
    r  = annual_return / 12
    n  = years * 12
    fv = monthly_sip * (((1 + r) ** n - 1) / r) * (1 + r)
    return round(fv, 2)


def horizon_to_years(horizon: str) -> int:
    return {"Short (1Y)": 1, "Medium (3Y)": 3, "Long (5Y+)": 5}.get(horizon, 3)


def _blended_sip(funds: list, monthly_sip: float, years: int) -> dict:
    n_funds      = len(funds) if funds else 1
    per_fund_sip = monthly_sip / n_funds

    total_projected = sum(
        sip_projection(per_fund_sip, f["metrics"]["ann_return"], years)
        for f in funds
    )
    total_invested = monthly_sip * 12 * years
    total_gain     = total_projected - total_invested
    gain_pct       = (total_gain / total_invested * 100) if total_invested > 0 else 0.0

    return {
        "per_fund_sip":    round(per_fund_sip, 0),
        "total_invested":  round(total_invested, 0),
        "total_projected": round(total_projected, 0),
        "total_gain":      round(total_gain, 0),
        "gain_pct":        round(gain_pct, 1),
    }


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def get_portfolio_recommendations(
    risk: str,
    horizon: str,
    monthly_sip: float,
    progress_callback=None,
) -> list:
    weights   = HORIZON_WEIGHTS.get(horizon, HORIZON_WEIGHTS["Medium (3Y)"])
    years     = horizon_to_years(horizon)
    templates = MODEL_PORTFOLIOS.get(risk, MODEL_PORTFOLIOS["Medium"])

    needed_categories: set = set()
    for t in templates:
        needed_categories.update(t["slots"])

    filtered_universe = [
        (code, name, cat)
        for code, name, cat in FUND_UNIVERSE_CLEAN
        if cat in needed_categories
    ]

    if not filtered_universe:
        return []

    all_metrics: list = []
    total = len(filtered_universe)

    for i, (code, name, cat) in enumerate(filtered_universe):
        if progress_callback:
            progress_callback(i + 1, total, name)
        m = _fetch_fund_metrics(code, horizon)
        if m is not None:
            m["fund_name"] = name
            m["category"]  = cat
            all_metrics.append(m)

    if not all_metrics:
        return []

    for m in all_metrics:
        m["score"] = _score_fund(m, all_metrics, weights)

    portfolios = []
    for template in templates:
        used_codes: set = set()
        funds = []

        for slot_idx, category in enumerate(template["slots"]):
            best = _pick_best_for_slot(category, all_metrics, used_codes)
            if best is None:
                continue
            used_codes.add(best["scheme_code"])
            funds.append({
                "scheme_code": best["scheme_code"],
                "fund_name":   best["fund_name"],
                "category":    best["category"],
                "metrics": {
                    "sharpe":         best["sharpe"],
                    "ann_return":     best["ann_return"],
                    "ann_volatility": best["ann_volatility"],
                    "max_drawdown":   best["max_drawdown"],
                    "horizon_return": best["horizon_return"],
                },
                "score":  best["score"],
                "reason": _fund_reason(best, best["category"], slot_idx),
            })

        if not funds:
            continue

        portfolios.append({
            "name":        template["name"],
            "description": template["description"],
            "emoji":       template["emoji"],
            "funds":       funds,
            "sip":         _blended_sip(funds, monthly_sip, years),
            "summary":     _portfolio_summary(template, risk, horizon),
        })

    return portfolios
