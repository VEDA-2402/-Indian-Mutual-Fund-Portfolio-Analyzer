"""
calculations/returns.py
-----------------------
Computes daily returns, rolling returns, and benchmark-relative returns.
No Streamlit imports — UI-agnostic.
"""

import pandas as pd
import numpy as np
from config import ROLLING_WINDOWS


# ---------------------------------------------------------------------------
# Daily Returns
# ---------------------------------------------------------------------------

def daily_returns(nav_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily percentage returns from NAV prices.
    Returns a DataFrame of the same shape (first row will be NaN, dropped).
    """
    return nav_df.pct_change().dropna()


def benchmark_daily_returns(benchmark: pd.Series) -> pd.Series:
    """
    Computes daily percentage returns for the benchmark series.
    """
    return benchmark.pct_change().dropna()


# ---------------------------------------------------------------------------
# Rolling Returns
# ---------------------------------------------------------------------------

def rolling_returns(nav_df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """
    Computes rolling returns over a given window (in trading days).
    Uses percentage change from `window_days` ago to today (not annualized).
    Returns a DataFrame aligned to nav_df's index.
    """
    return nav_df.pct_change(periods=window_days).dropna()


def all_rolling_returns(nav_df: pd.DataFrame) -> dict:
    """
    Computes rolling returns for all windows defined in config.ROLLING_WINDOWS.
    Returns a dict: { '1M': DataFrame, '3M': DataFrame, ... }
    """
    result = {}
    for label, days in ROLLING_WINDOWS.items():
        if len(nav_df) > days:
            result[label] = rolling_returns(nav_df, days)
        else:
            result[label] = pd.DataFrame()  # not enough data for this window
    return result


def latest_rolling_returns(nav_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary DataFrame of the most recent rolling return for each
    window and each fund.
    Shape: rows = windows (1M, 3M, ...), columns = fund scheme codes.
    Values are percentage returns (e.g. 0.12 = 12%).
    """
    records = {}
    for label, days in ROLLING_WINDOWS.items():
        if len(nav_df) > days:
            row = nav_df.pct_change(periods=days).iloc[-1]
            records[label] = row
        else:
            records[label] = pd.Series({col: np.nan for col in nav_df.columns})

    return pd.DataFrame(records).T  # rows = windows, cols = funds


# ---------------------------------------------------------------------------
# Cumulative Returns
# ---------------------------------------------------------------------------

def cumulative_returns(nav_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cumulative returns from the start of the series.
    Formula: (NAV_t / NAV_0) - 1
    Returns a DataFrame with same index as nav_df.
    """
    return (nav_df / nav_df.iloc[0]) - 1


# ---------------------------------------------------------------------------
# Annualized Returns
# ---------------------------------------------------------------------------

def annualized_return(nav_df: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """
    Computes annualized return for each fund over the full available history.
    Formula: (final_NAV / initial_NAV) ^ (252 / n_days) - 1
    Returns a pd.Series indexed by fund scheme code.
    """
    n = len(nav_df)
    if n < 2:
        return pd.Series({col: np.nan for col in nav_df.columns})

    total_return = nav_df.iloc[-1] / nav_df.iloc[0]
    ann_return = total_return ** (trading_days / n) - 1
    return ann_return


# ---------------------------------------------------------------------------
# Benchmark Comparison
# ---------------------------------------------------------------------------

def benchmark_comparison(nav_df: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    """
    Computes rolling returns for each fund AND benchmark for all windows.
    Returns a dict: { window_label: DataFrame with fund cols + 'Nifty50' col }
    Useful for direct side-by-side comparison charts.
    """
    result = {}
    bench_df = benchmark.to_frame()  # Series → single-col DataFrame

    for label, days in ROLLING_WINDOWS.items():
        if len(nav_df) > days:
            fund_roll  = nav_df.pct_change(periods=days).dropna()
            bench_roll = bench_df.pct_change(periods=days).dropna()

            # Align on common index
            common_idx = fund_roll.index.intersection(bench_roll.index)
            combined   = pd.concat(
                [fund_roll.loc[common_idx], bench_roll.loc[common_idx]],
                axis=1
            )
            result[label] = combined
        else:
            result[label] = pd.DataFrame()

    return result


def excess_returns(nav_df: pd.DataFrame, benchmark: pd.Series) -> pd.Series:
    """
    Computes annualized excess return of each fund over the benchmark
    over the full period.
    Returns a pd.Series: fund_annualized_return - benchmark_annualized_return
    """
    fund_ann  = annualized_return(nav_df)
    bench_ann = annualized_return(benchmark.to_frame())

    bench_val = bench_ann.iloc[0] if not bench_ann.empty else np.nan
    return fund_ann - bench_val


# ---------------------------------------------------------------------------
# SIP Calculator
# ---------------------------------------------------------------------------

def _xirr(cashflows: list) -> float:
    """
    Computes XIRR given a list of (date, amount) tuples.
    Negative amount = investment (outflow), positive = redemption (inflow).
    Uses Newton-Raphson iteration. Returns annualised rate or np.nan on failure.
    """
    from datetime import date as date_type

    if len(cashflows) < 2:
        return np.nan

    dates   = [cf[0] for cf in cashflows]
    amounts = [cf[1] for cf in cashflows]
    t0      = dates[0]

    def npv(rate):
        return sum(
            amt / (1 + rate) ** ((d - t0).days / 365.0)
            for d, amt in zip(dates, amounts)
        )

    def dnpv(rate):
        return sum(
            -((d - t0).days / 365.0) * amt / (1 + rate) ** ((d - t0).days / 365.0 + 1)
            for d, amt in zip(dates, amounts)
        )

    rate = 0.1  # initial guess
    for _ in range(200):
        try:
            f  = npv(rate)
            df = dnpv(rate)
            if df == 0:
                break
            new_rate = rate - f / df
            if abs(new_rate - rate) < 1e-6:
                return round(new_rate, 6)
            rate = new_rate
        except Exception:
            break
    return np.nan


def sip_analysis(nav_df: pd.DataFrame, monthly_amount: float) -> dict:
    """
    Simulates a monthly SIP for each fund using actual historical NAVs.

    For each fund:
      - On the 1st of every month (or nearest available NAV date), invest `monthly_amount`
      - Units accumulated = amount / NAV on that date
      - Current value = total units × latest NAV
      - XIRR = annualised return accounting for timing of each installment

    Parameters
    ----------
    nav_df         : DataFrame of NAV prices (DatetimeIndex, cols = scheme codes)
    monthly_amount : float — monthly SIP amount in ₹ per fund

    Returns
    -------
    dict with keys:
      'corpus_df'     : DataFrame — monthly portfolio value per fund over time
      'invested_df'   : DataFrame — cumulative amount invested per fund over time
      'summary'       : DataFrame — total invested, final value, gain, XIRR per fund
      'best_fund'     : str — scheme code with highest XIRR
      'monthly_amount': float
    """
    results      = {}
    corpus_data  = {}
    invested_data = {}

    for col in nav_df.columns:
        series = nav_df[col].dropna()
        if series.empty:
            continue

        # Generate monthly SIP dates (1st of each month in the NAV range)
        sip_months = pd.date_range(
            start=series.index.min().replace(day=1),
            end=series.index.max().replace(day=1),
            freq="MS"  # Month Start
        )

        total_units    = 0.0
        total_invested = 0.0
        cashflows      = []   # for XIRR: (date, amount)
        corpus_series  = {}
        invested_series = {}

        for sip_date in sip_months:
            # Find nearest available NAV on or after sip_date
            available = series[series.index >= sip_date]
            if available.empty:
                continue

            nav_date = available.index[0]
            nav_val  = available.iloc[0]

            units           = monthly_amount / nav_val
            total_units    += units
            total_invested += monthly_amount

            cashflows.append((nav_date.date(), -monthly_amount))  # outflow

            # Portfolio value on this date
            corpus_series[nav_date]   = total_units * nav_val
            invested_series[nav_date] = total_invested

        if not corpus_series:
            continue

        # Final redemption cashflow (positive inflow = current value)
        latest_nav  = series.iloc[-1]
        latest_date = series.index[-1].date()
        final_value = total_units * latest_nav
        cashflows.append((latest_date, final_value))

        xirr_val = _xirr(cashflows)

        gain     = final_value - total_invested
        gain_pct = (gain / total_invested * 100) if total_invested > 0 else 0.0

        results[col] = {
            "total_invested": round(total_invested, 2),
            "final_value":    round(final_value, 2),
            "gain":           round(gain, 2),
            "gain_pct":       round(gain_pct, 2),
            "xirr":           xirr_val,
            "total_units":    round(total_units, 4),
            "latest_nav":     round(latest_nav, 4),
            "sip_count":      len([c for c in cashflows if c[1] < 0]),
        }
        corpus_data[col]   = pd.Series(corpus_series)
        invested_data[col] = pd.Series(invested_series)

    if not results:
        raise RuntimeError("SIP simulation failed — no valid NAV data for any fund.")

    corpus_df   = pd.DataFrame(corpus_data).sort_index().ffill()
    invested_df = pd.DataFrame(invested_data).sort_index().ffill()
    summary_df  = pd.DataFrame(results).T

    best_fund = summary_df["xirr"].idxmax() if not summary_df["xirr"].isna().all() else None

    return {
        "corpus_df":      corpus_df,
        "invested_df":    invested_df,
        "summary":        summary_df,
        "best_fund":      best_fund,
        "monthly_amount": monthly_amount,
    }
