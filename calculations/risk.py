"""
calculations/risk.py
--------------------
Computes risk metrics: volatility, Sharpe ratio, max drawdown, and correlation.
No Streamlit imports — UI-agnostic.
"""

import pandas as pd
import numpy as np
from config import RISK_FREE_RATE


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def annualized_volatility(daily_returns_df: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """
    Computes annualized volatility (std dev of daily returns * sqrt(252)).
    Input: DataFrame of daily returns (from calculations/returns.py).
    Returns: pd.Series indexed by fund scheme code.
    """
    return daily_returns_df.std() * np.sqrt(trading_days)


def rolling_volatility(daily_returns_df: pd.DataFrame, window: int = 30, trading_days: int = 252) -> pd.DataFrame:
    """
    Computes rolling annualized volatility over a given window.
    Useful for plotting how risk evolves over time.
    Returns: DataFrame with same columns as input.
    """
    return daily_returns_df.rolling(window=window).std() * np.sqrt(trading_days)


# ---------------------------------------------------------------------------
# Sharpe Ratio
# ---------------------------------------------------------------------------

def sharpe_ratio(daily_returns_df: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """
    Computes annualized Sharpe Ratio for each fund.
    Formula: (annualized_return - risk_free_rate) / annualized_volatility
    Risk-free rate from config (default 6% = Indian T-bill proxy).
    Returns: pd.Series indexed by fund scheme code.
    """
    mean_daily = daily_returns_df.mean()
    std_daily  = daily_returns_df.std()

    # Avoid division by zero
    std_daily = std_daily.replace(0, np.nan)

    ann_return = mean_daily * trading_days
    ann_vol    = std_daily  * np.sqrt(trading_days)

    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol
    return sharpe


# ---------------------------------------------------------------------------
# Maximum Drawdown
# ---------------------------------------------------------------------------

def max_drawdown(nav_df: pd.DataFrame) -> pd.Series:
    """
    Computes Maximum Drawdown (MDD) for each fund.
    Formula: (peak - trough) / peak — worst peak-to-trough decline.
    Returns: pd.Series of negative values indexed by fund scheme code.
             e.g. -0.32 means a 32% drawdown.
    """
    result = {}
    for col in nav_df.columns:
        series      = nav_df[col].dropna()
        rolling_max = series.cummax()
        drawdown    = (series - rolling_max) / rolling_max
        result[col] = drawdown.min()  # most negative value = max drawdown

    return pd.Series(result)


def drawdown_series(nav_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the full drawdown time series for each fund.
    Useful for plotting the drawdown chart over time.
    Returns: DataFrame with same index as nav_df, values are drawdown fractions.
    """
    result = {}
    for col in nav_df.columns:
        series      = nav_df[col].dropna()
        rolling_max = series.cummax()
        result[col] = (series - rolling_max) / rolling_max

    return pd.DataFrame(result)


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def correlation_matrix(daily_returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the Pearson correlation matrix of daily returns between all funds.
    Returns: Square DataFrame (funds x funds), values in [-1, 1].
    """
    return daily_returns_df.corr(method="pearson")


def average_pairwise_correlation(corr_matrix: pd.DataFrame) -> float:
    """
    Computes the average pairwise correlation across all fund pairs (excluding diagonal).
    Used as an input to the Portfolio Health Score.
    Returns: float in [-1, 1]. Higher = more correlated = less diversified.
    """
    n = len(corr_matrix)
    if n < 2:
        return 0.0

    # Extract upper triangle (excluding diagonal)
    mask  = np.triu(np.ones((n, n), dtype=bool), k=1)
    upper = corr_matrix.values[mask]

    return float(np.nanmean(upper))


# ---------------------------------------------------------------------------
# Summary Risk Table
# ---------------------------------------------------------------------------

def risk_summary(nav_df: pd.DataFrame, daily_returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines all risk metrics into a single summary DataFrame.
    Rows = funds, Columns = [Volatility, Sharpe, MaxDrawdown].
    Useful for the scorecard table in the UI.
    """
    vol      = annualized_volatility(daily_returns_df).rename("Volatility (Ann.)")
    sharpe   = sharpe_ratio(daily_returns_df).rename("Sharpe Ratio")
    mdd      = max_drawdown(nav_df).rename("Max Drawdown")

    summary = pd.concat([vol, sharpe, mdd], axis=1)
    return summary
