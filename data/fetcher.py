"""
data/fetcher.py
---------------
Fetches mutual fund NAV data (via mftool) and Nifty 50 benchmark data (via yfinance).
Uses hybrid disk cache (data/cache_manager.py) — auto-expires after 24h.
No Streamlit imports — this module is UI-agnostic and cloud-deployable.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple
from mftool import Mftool
import yfinance as yf

from config import BENCHMARK_TICKER
from data.cache_manager import (
    get_cached_fund_list, save_fund_list,
    get_cached_nav, save_nav,
    get_cached_benchmark, save_benchmark,
)

# --- Module-level mftool instance ---
_mf = Mftool()


# ---------------------------------------------------------------------------
# Fund Search / Listing
# ---------------------------------------------------------------------------

def get_all_funds() -> dict:
    """
    Returns a dict of {scheme_code: scheme_name} for all funds available via AMFI.
    Checks local cache first (24h TTL). Falls back to live AMFI fetch if stale/missing.
    """
    cached = get_cached_fund_list()
    if cached:
        return cached

    try:
        funds = _mf.get_scheme_codes()
        if not funds:
            raise ValueError("Empty fund list returned from mftool.")
        save_fund_list(funds)
        return funds
    except Exception as e:
        raise RuntimeError(f"Failed to fetch fund list: {e}")


def search_funds_by_name(query: str) -> dict:
    """
    Filters the full fund list by a name substring (case-insensitive).
    Returns {scheme_code: scheme_name}.
    """
    all_funds = get_all_funds()
    query_lower = query.lower()
    return {
        code: name
        for code, name in all_funds.items()
        if query_lower in name.lower()
    }


# ---------------------------------------------------------------------------
# NAV Data
# ---------------------------------------------------------------------------

def get_nav_history(scheme_code: str) -> pd.Series:
    """
    Fetches full NAV history for a given scheme code.
    Checks local cache first (24h TTL). Falls back to live AMFI fetch if stale/missing.
    Returns a pd.Series with DatetimeIndex, sorted ascending, named by scheme_code.
    Raises RuntimeError on failure.
    """
    cached = get_cached_nav(scheme_code)
    if cached is not None:
        return cached

    try:
        raw = _mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)

        if raw is None or raw.empty:
            raise ValueError(f"No NAV data returned for scheme {scheme_code}.")

        # mftool returns columns: ['date', 'nav'] or index may vary — normalize
        raw = raw.reset_index()

        # Identify date and nav columns (case-insensitive)
        col_map = {c.lower(): c for c in raw.columns}
        date_col = col_map.get("date")
        nav_col  = col_map.get("nav")

        if not date_col or not nav_col:
            raise ValueError(f"Unexpected columns from mftool: {list(raw.columns)}")

        raw[date_col] = pd.to_datetime(raw[date_col], format="%d-%m-%Y", errors="coerce")
        raw[nav_col]  = pd.to_numeric(raw[nav_col], errors="coerce")

        series = (
            raw[[date_col, nav_col]]
            .dropna()
            .set_index(date_col)[nav_col]
            .sort_index()
            .rename(scheme_code)
        )

        if series.empty:
            raise ValueError(f"NAV series is empty after cleaning for scheme {scheme_code}.")

        save_nav(scheme_code, series)
        return series

    except Exception as e:
        raise RuntimeError(f"Failed to fetch NAV for scheme {scheme_code}: {e}")


def get_multiple_nav(scheme_codes: list) -> pd.DataFrame:
    """
    Fetches NAV history for multiple schemes and aligns them on a common date index.
    Returns a DataFrame where each column is a scheme_code.
    Missing values are forward-filled (fund holidays), then dropped.
    """
    series_list = []
    errors = {}

    for code in scheme_codes:
        try:
            s = get_nav_history(code)
            series_list.append(s)
        except RuntimeError as e:
            errors[code] = str(e)

    if errors:
        error_msg = "\n".join([f"  {k}: {v}" for k, v in errors.items()])
        raise RuntimeError(f"Failed to fetch NAV for one or more funds:\n{error_msg}")

    if not series_list:
        raise RuntimeError("No NAV data could be fetched for any of the selected funds.")

    df = pd.concat(series_list, axis=1)
    df = df.sort_index()
    df = df.ffill()   # forward fill for non-trading days
    df = df.dropna()  # drop rows where any fund has no data

    return df


# ---------------------------------------------------------------------------
# Benchmark (Nifty 50)
# ---------------------------------------------------------------------------

def get_benchmark_data(start_date: str = None, end_date: str = None) -> pd.Series:
    """
    Fetches Nifty 50 closing price history via yfinance.
    Checks local cache first (24h TTL). On cache hit, slices to requested date range.
    start_date / end_date: 'YYYY-MM-DD' strings. Defaults to last 5 years.
    Returns a pd.Series with DatetimeIndex, named 'Nifty50'.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

    cached = get_cached_benchmark()
    if cached is not None:
        sliced = cached[cached.index >= pd.Timestamp(start_date)]
        if not sliced.empty:
            return sliced

    try:
        # Always fetch max history so cache stays useful across date ranges
        fetch_start = (datetime.today() - timedelta(days=10 * 365)).strftime("%Y-%m-%d")
        ticker = yf.Ticker(BENCHMARK_TICKER)
        df = ticker.history(start=fetch_start, end=end_date)

        if df is None or df.empty:
            raise ValueError(f"No data returned from yfinance for {BENCHMARK_TICKER}.")

        series = df["Close"].rename("Nifty50")
        series.index = pd.to_datetime(series.index).tz_localize(None)
        series = series.sort_index()

        save_benchmark(series)

        # Return only the requested slice
        return series[series.index >= pd.Timestamp(start_date)]

    except Exception as e:
        raise RuntimeError(f"Failed to fetch benchmark data ({BENCHMARK_TICKER}): {e}")


# ---------------------------------------------------------------------------
# Fund Category / Type Info
# ---------------------------------------------------------------------------

def get_fund_info(scheme_codes: list) -> dict:
    """
    Fetches scheme_category and scheme_type for each fund via mftool.get_scheme_details().
    Returns dict: {scheme_code: {'category': str, 'fund_type': str, 'fund_house': str}}
    Silently skips any code that fails — callers should handle missing keys gracefully.
    """
    result = {}
    for code in scheme_codes:
        try:
            details = _mf.get_scheme_details(str(code))
            if details:
                result[code] = {
                    "category":  details.get("scheme_category", "N/A"),
                    "fund_type": details.get("scheme_type", "N/A"),
                    "fund_house": details.get("fund_house", "N/A"),
                }
        except Exception:
            result[code] = {"category": "N/A", "fund_type": "N/A", "fund_house": "N/A"}
    return result


# ---------------------------------------------------------------------------
# NAV Snapshot — Today's NAV + 52-week High/Low
# ---------------------------------------------------------------------------

def get_nav_snapshot(nav_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a snapshot table from the already-fetched NAV DataFrame.
    No extra API calls — derived entirely from nav_df.

    Returns a DataFrame with columns:
      latest_nav, latest_date, week52_high, week52_low,
      pct_from_high, pct_from_low, week52_high_date, week52_low_date

    Index = scheme codes (same as nav_df columns).
    """
    cutoff_52w = pd.Timestamp.today() - pd.DateOffset(weeks=52)
    records = {}

    for col in nav_df.columns:
        series = nav_df[col].dropna()
        if series.empty:
            continue

        latest_nav  = float(series.iloc[-1])
        latest_date = series.index[-1]

        # 52-week window
        series_52w = series[series.index >= cutoff_52w]
        if series_52w.empty:
            series_52w = series  # fallback: use full history if < 1 year of data

        high_val  = float(series_52w.max())
        low_val   = float(series_52w.min())
        high_date = series_52w.idxmax()
        low_date  = series_52w.idxmin()

        pct_from_high = ((latest_nav - high_val) / high_val) * 100  # negative = below peak
        pct_from_low  = ((latest_nav - low_val)  / low_val)  * 100  # positive = above trough

        records[col] = {
            "latest_nav":       round(latest_nav, 4),
            "latest_date":      latest_date.strftime("%d %b %Y"),
            "week52_high":      round(high_val, 4),
            "week52_high_date": high_date.strftime("%d %b %Y"),
            "week52_low":       round(low_val, 4),
            "week52_low_date":  low_date.strftime("%d %b %Y"),
            "pct_from_high":    round(pct_from_high, 2),
            "pct_from_low":     round(pct_from_low, 2),
        }

    return pd.DataFrame(records).T  # rows = funds


# ---------------------------------------------------------------------------
# Aligned Portfolio + Benchmark
# ---------------------------------------------------------------------------

def get_aligned_data(scheme_codes: list) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Fetches fund NAVs and Nifty 50, aligns them on overlapping dates.
    Returns (nav_df, benchmark_series) both on the same DatetimeIndex.
    """
    nav_df = get_multiple_nav(scheme_codes)

    start_date = nav_df.index.min().strftime("%Y-%m-%d")
    end_date   = nav_df.index.max().strftime("%Y-%m-%d")

    benchmark = get_benchmark_data(start_date=start_date, end_date=end_date)

    # Align on common dates
    common_idx = nav_df.index.intersection(benchmark.index)
    if len(common_idx) < 30:
        raise RuntimeError(
            "Insufficient overlapping dates between fund NAV and Nifty 50 data. "
            "Try selecting funds with longer history."
        )

    nav_df    = nav_df.loc[common_idx]
    benchmark = benchmark.loc[common_idx]

    return nav_df, benchmark
