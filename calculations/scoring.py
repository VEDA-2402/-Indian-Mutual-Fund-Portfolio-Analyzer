"""
calculations/scoring.py
------------------------
Computes the Portfolio Health Score (0-100) with a breakdown by category.
Scoring logic is fully deterministic and UI-agnostic.
"""

import numpy as np
import pandas as pd
from config import SCORE_WEIGHTS


# ---------------------------------------------------------------------------
# Individual Component Scorers (each returns 0–100)
# ---------------------------------------------------------------------------

def _score_sharpe(avg_sharpe: float) -> float:
    """
    Converts average Sharpe ratio across funds to a 0–100 score.
    Thresholds (typical for Indian equity MFs):
      >= 1.5  → 100  (excellent)
      1.0–1.5 → 75   (good)
      0.5–1.0 → 50   (average)
      0.0–0.5 → 25   (below average)
      < 0.0   → 0    (poor / losing against risk-free)
    """
    if np.isnan(avg_sharpe):
        return 0.0
    if avg_sharpe >= 1.5:
        return 100.0
    elif avg_sharpe >= 1.0:
        return 75.0 + (avg_sharpe - 1.0) / 0.5 * 25.0
    elif avg_sharpe >= 0.5:
        return 50.0 + (avg_sharpe - 0.5) / 0.5 * 25.0
    elif avg_sharpe >= 0.0:
        return 25.0 + avg_sharpe / 0.5 * 25.0
    else:
        return max(0.0, 25.0 + avg_sharpe * 25.0)  # linearly penalize negative Sharpe


def _score_drawdown(avg_max_drawdown: float) -> float:
    """
    Converts average max drawdown (negative fraction) to a 0–100 score.
    Lower drawdown = higher score.
    Thresholds:
      0% to -10%  → 100–75  (low risk)
      -10% to -20% → 75–50  (moderate)
      -20% to -35% → 50–25  (high)
      -35% to -50% → 25–0   (very high)
      < -50%       → 0      (extreme)
    """
    if np.isnan(avg_max_drawdown):
        return 0.0

    dd = abs(avg_max_drawdown)  # work with positive magnitude

    if dd <= 0.10:
        return 100.0 - (dd / 0.10) * 25.0
    elif dd <= 0.20:
        return 75.0 - ((dd - 0.10) / 0.10) * 25.0
    elif dd <= 0.35:
        return 50.0 - ((dd - 0.20) / 0.15) * 25.0
    elif dd <= 0.50:
        return 25.0 - ((dd - 0.35) / 0.15) * 25.0
    else:
        return 0.0


def _score_correlation(avg_pairwise_corr: float) -> float:
    """
    Converts average pairwise correlation to a 0–100 score.
    Lower correlation = better diversification = higher score.
    Thresholds:
      <= 0.3  → 100  (well diversified)
      0.3–0.6 → 75–50 (moderate overlap)
      0.6–0.8 → 50–25 (high overlap)
      > 0.8   → 0–25  (very concentrated)

    Note: If only 1 fund is selected, correlation is undefined → returns 100
    (no penalty for single fund, no diversification assessment possible).
    """
    if np.isnan(avg_pairwise_corr):
        return 100.0  # single fund case

    corr = avg_pairwise_corr

    if corr <= 0.3:
        return 100.0
    elif corr <= 0.6:
        return 100.0 - ((corr - 0.3) / 0.3) * 50.0
    elif corr <= 0.8:
        return 50.0 - ((corr - 0.6) / 0.2) * 25.0
    else:
        return max(0.0, 25.0 - ((corr - 0.8) / 0.2) * 25.0)


def _score_returns(avg_1y_return: float) -> float:
    """
    Converts average 1-year rolling return to a 0–100 score.
    Thresholds (Indian equity MF benchmarks):
      >= 20%  → 100  (excellent)
      15–20%  → 80   (very good)
      10–15%  → 60   (good — beats inflation)
      5–10%   → 40   (average)
      0–5%    → 20   (poor)
      < 0%    → 0    (negative returns)
    """
    if np.isnan(avg_1y_return):
        return 0.0

    r = avg_1y_return

    if r >= 0.20:
        return 100.0
    elif r >= 0.15:
        return 80.0 + ((r - 0.15) / 0.05) * 20.0
    elif r >= 0.10:
        return 60.0 + ((r - 0.10) / 0.05) * 20.0
    elif r >= 0.05:
        return 40.0 + ((r - 0.05) / 0.05) * 20.0
    elif r >= 0.0:
        return 20.0 + (r / 0.05) * 20.0
    else:
        return max(0.0, 20.0 + r * 100.0)  # linearly penalize negative returns


# ---------------------------------------------------------------------------
# Portfolio Health Score
# ---------------------------------------------------------------------------

def portfolio_health_score(
    sharpe_series: pd.Series,
    drawdown_series: pd.Series,
    avg_pairwise_corr: float,
    latest_rolling_returns: pd.DataFrame,
) -> dict:
    """
    Computes the overall Portfolio Health Score (0–100) and a full breakdown.

    Parameters
    ----------
    sharpe_series         : pd.Series — Sharpe ratio per fund
    drawdown_series       : pd.Series — Max drawdown per fund (negative fractions)
    avg_pairwise_corr     : float     — Average pairwise correlation across funds
    latest_rolling_returns: pd.DataFrame — rows=windows, cols=funds (from returns.py)

    Returns
    -------
    dict with keys:
      'total_score'      : float (0–100)
      'grade'            : str ('A', 'B', 'C', 'D', 'F')
      'components'       : dict of component scores and raw values
      'weights'          : dict of weights used
      'interpretation'   : str — plain English summary
    """

    # --- Raw values ---
    avg_sharpe  = float(sharpe_series.mean())   if not sharpe_series.empty  else np.nan
    avg_mdd     = float(drawdown_series.mean()) if not drawdown_series.empty else np.nan

    # 1Y return: use latest_rolling_returns row if available, else NaN
    if "1Y" in latest_rolling_returns.index and not latest_rolling_returns.loc["1Y"].empty:
        avg_1y_return = float(latest_rolling_returns.loc["1Y"].mean())
    else:
        avg_1y_return = np.nan

    # --- Component scores (0–100 each) ---
    s_sharpe  = _score_sharpe(avg_sharpe)
    s_dd      = _score_drawdown(avg_mdd)
    s_corr    = _score_correlation(avg_pairwise_corr)
    s_returns = _score_returns(avg_1y_return)

    # --- Weighted total ---
    w = SCORE_WEIGHTS
    total = (
        s_sharpe  * w["sharpe"]      +
        s_dd      * w["drawdown"]    +
        s_corr    * w["correlation"] +
        s_returns * w["returns"]
    )
    total = round(min(100.0, max(0.0, total)), 1)

    # --- Grade ---
    if total >= 80:
        grade = "A"
    elif total >= 65:
        grade = "B"
    elif total >= 50:
        grade = "C"
    elif total >= 35:
        grade = "D"
    else:
        grade = "F"

    # --- Interpretation ---
    interpretation = _interpret(total, grade, s_corr, s_dd, s_sharpe)

    return {
        "total_score": total,
        "grade": grade,
        "components": {
            "Sharpe Ratio": {
                "score": round(s_sharpe, 1),
                "raw_value": round(avg_sharpe, 3) if not np.isnan(avg_sharpe) else None,
                "label": "Risk-adjusted return quality"
            },
            "Max Drawdown": {
                "score": round(s_dd, 1),
                "raw_value": f"{avg_mdd * 100:.1f}%" if not np.isnan(avg_mdd) else None,
                "label": "Downside risk protection"
            },
            "Diversification": {
                "score": round(s_corr, 1),
                "raw_value": round(avg_pairwise_corr, 3) if not np.isnan(avg_pairwise_corr) else None,
                "label": "Inter-fund correlation (lower = better)"
            },
            "Consistent Returns": {
                "score": round(s_returns, 1),
                "raw_value": f"{avg_1y_return * 100:.1f}%" if not np.isnan(avg_1y_return) else None,
                "label": "1-Year rolling return quality"
            },
        },
        "weights": w,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Individual Fund Score
# ---------------------------------------------------------------------------

def individual_fund_score(
    scheme_code: str,
    sharpe_series: pd.Series,
    drawdown_series: pd.Series,
    latest_rolling_returns: pd.DataFrame,
) -> dict:
    """
    Computes a Health Score for a single fund (no correlation component —
    correlation is a portfolio-level concept, so it's excluded here and
    its weight redistributed to other components).

    Returns same structure as portfolio_health_score() for UI reuse.
    """
    sharpe_val = float(sharpe_series.get(scheme_code, np.nan))
    mdd_val    = float(drawdown_series.get(scheme_code, np.nan))

    if "1Y" in latest_rolling_returns.index:
        ret_val = float(latest_rolling_returns.loc["1Y"].get(scheme_code, np.nan))
    else:
        ret_val = np.nan

    s_sharpe  = _score_sharpe(sharpe_val)
    s_dd      = _score_drawdown(mdd_val)
    s_returns = _score_returns(ret_val)

    # Redistribute correlation weight (20%) equally to other 3 components
    w_sharpe  = SCORE_WEIGHTS["sharpe"]  + SCORE_WEIGHTS["correlation"] / 3
    w_dd      = SCORE_WEIGHTS["drawdown"]+ SCORE_WEIGHTS["correlation"] / 3
    w_returns = SCORE_WEIGHTS["returns"] + SCORE_WEIGHTS["correlation"] / 3

    total = s_sharpe * w_sharpe + s_dd * w_dd + s_returns * w_returns
    total = round(min(100.0, max(0.0, total)), 1)

    if total >= 80:   grade = "A"
    elif total >= 65: grade = "B"
    elif total >= 50: grade = "C"
    elif total >= 35: grade = "D"
    else:             grade = "F"

    interpretation = _interpret(total, grade, s_corr=100, s_dd=s_dd, s_sharpe=s_sharpe)

    return {
        "total_score": total,
        "grade": grade,
        "scheme_code": scheme_code,
        "components": {
            "Sharpe Ratio": {
                "score": round(s_sharpe, 1),
                "raw_value": round(sharpe_val, 3) if not np.isnan(sharpe_val) else None,
                "label": "Risk-adjusted return quality"
            },
            "Max Drawdown": {
                "score": round(s_dd, 1),
                "raw_value": f"{mdd_val * 100:.1f}%" if not np.isnan(mdd_val) else None,
                "label": "Downside risk protection"
            },
            "Consistent Returns": {
                "score": round(s_returns, 1),
                "raw_value": f"{ret_val * 100:.1f}%" if not np.isnan(ret_val) else None,
                "label": "1-Year rolling return quality"
            },
        },
        "weights": {"sharpe": w_sharpe, "drawdown": w_dd, "returns": w_returns},
        "interpretation": interpretation,
    }


def all_individual_scores(
    sharpe_series: pd.Series,
    drawdown_series: pd.Series,
    latest_rolling_returns: pd.DataFrame,
) -> list:
    """
    Returns a list of individual score dicts for every fund in sharpe_series.
    Sorted descending by total_score.
    """
    scores = [
        individual_fund_score(code, sharpe_series, drawdown_series, latest_rolling_returns)
        for code in sharpe_series.index
    ]
    return sorted(scores, key=lambda x: x["total_score"], reverse=True)


def _interpret(total: float, grade: str, s_corr: float, s_dd: float, s_sharpe: float) -> str:
    """Generates a plain-English summary of the portfolio health."""
    lines = []

    if grade == "A":
        lines.append("Your portfolio is in excellent shape with strong risk-adjusted returns.")
    elif grade == "B":
        lines.append("Your portfolio is performing well with room for minor improvements.")
    elif grade == "C":
        lines.append("Your portfolio is average — consider reviewing underperforming areas.")
    elif grade == "D":
        lines.append("Your portfolio has significant weaknesses that need attention.")
    else:
        lines.append("Your portfolio is underperforming across most dimensions.")

    if s_corr < 40:
        lines.append("High inter-fund correlation is reducing your diversification benefit.")
    if s_dd < 40:
        lines.append("Large drawdowns indicate high downside risk in your portfolio.")
    if s_sharpe < 40:
        lines.append("Low Sharpe ratio suggests returns are not adequately compensating for risk.")

    return " ".join(lines)
