"""
app.py
------
Main Streamlit entry point for the Indian Mutual Fund Portfolio Analyzer.
Wires together: data fetching → calculations → UI rendering.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from config import APP_TITLE, APP_ICON
from data.fetcher import get_all_funds, get_aligned_data, get_nav_snapshot, get_fund_info
from data.cache_manager import clear_all_cache, get_cache_status
from calculations.returns import (
    daily_returns,
    cumulative_returns,
    all_rolling_returns,
    latest_rolling_returns,
    benchmark_comparison,
    excess_returns,
    annualized_return,
)
from calculations.risk import (
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    drawdown_series,
    correlation_matrix,
    average_pairwise_correlation,
    risk_summary,
)
from calculations.scoring import portfolio_health_score, all_individual_scores
from ui.components import (
    set_page_config,
    render_fund_selector,
    render_date_range,
    render_summary_card,
    render_nav_snapshot,
    render_metric_cards,
    render_rolling_returns_table,
    render_benchmark_table,
    render_health_scorecard,
    render_charts_section,
    render_sip_section,
    render_error,
    render_loading_placeholder,
)

# ---------------------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------------------

set_page_config(APP_TITLE, APP_ICON)

# --- Header: Logo + Title side by side ---
logo_col, title_col = st.columns([1, 5])
with logo_col:
    try:
        st.image("logo/LOGO_PORTFOLIO_WEBSITE.png", use_container_width=True)
    except Exception:
        pass  # silently skip if logo not found
with title_col:
    st.title(f"{APP_TITLE}")
    st.caption("Analyze your mutual fund portfolio — returns, risk, correlation, and health score.")

st.divider()


# ---------------------------------------------------------------------------
# Load Fund List (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Loading fund list...")
def load_fund_list():
    return get_all_funds()


@st.cache_data(ttl=3600, show_spinner="Fetching NAV and benchmark data...")
def load_portfolio_data(scheme_codes: tuple):
    """Tuple input ensures hashability for Streamlit cache."""
    return get_aligned_data(list(scheme_codes))


# ---------------------------------------------------------------------------
# Sidebar — Fund Selector & Period
# ---------------------------------------------------------------------------

# --- Cache Controls ---
st.sidebar.divider()
st.sidebar.header("🔄 Data Cache")

cache_status = get_cache_status()
if cache_status:
    fund_status = cache_status.get("fund_list", {})
    if fund_status:
        freshness = "🟢 Fresh" if fund_status["is_fresh"] else "🔴 Expired"
        st.sidebar.caption(
            f"{freshness} — Last updated: {fund_status['last_fetched']}\n"
            f"Expires in: {fund_status['expires_in_h']}h"
        )
else:
    st.sidebar.caption("No cache yet — data will be fetched fresh.")

if st.sidebar.button("🔃 Refresh All Data", help="Clears local cache and re-fetches everything from AMFI & yfinance"):
    clear_all_cache()
    st.cache_data.clear()
    st.sidebar.success("Cache cleared! Reloading...")
    st.rerun()

st.sidebar.divider()

try:
    all_funds = load_fund_list()
except RuntimeError as e:
    render_error(f"Could not load fund list. Please check your internet connection.\n\nDetails: {e}")
    st.stop()

selected_codes = render_fund_selector(all_funds)
period_label, lookback_years = render_date_range()

# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

if not selected_codes:
    render_loading_placeholder()
    st.stop()

# --- Fetch Data ---
with st.spinner("Fetching data for selected funds..."):
    try:
        nav_df, benchmark = load_portfolio_data(tuple(selected_codes))
    except RuntimeError as e:
        render_error(str(e))
        st.stop()

# --- Apply Period Filter ---
if lookback_years is not None:
    cutoff = nav_df.index.max() - pd.DateOffset(years=lookback_years)
    nav_df    = nav_df[nav_df.index >= cutoff]
    benchmark = benchmark[benchmark.index >= cutoff]

if len(nav_df) < 30:
    render_error(
        "Not enough data for the selected period. "
        "Try selecting a longer period or different funds."
    )
    st.stop()

# --- Build fund name lookup (code → name) ---
fund_names = {code: all_funds.get(code, code) for code in selected_codes}

# --- Fetch fund category/type info (cached via mftool, non-blocking) ---
with st.spinner("Loading fund details..."):
    fund_info = get_fund_info(selected_codes)  # {code: {category, fund_type, fund_house}}

# ---------------------------------------------------------------------------
# Calculations
# ---------------------------------------------------------------------------

daily_ret_df       = daily_returns(nav_df)
cumul_df           = cumulative_returns(nav_df)
rolling_all        = all_rolling_returns(nav_df)
latest_rolling_df  = latest_rolling_returns(nav_df)
rolling_vs_bench   = benchmark_comparison(nav_df, benchmark)
excess_ret         = excess_returns(nav_df, benchmark)
risk_summary_df    = risk_summary(nav_df, daily_ret_df)
drawdown_df        = drawdown_series(nav_df)
corr_matrix        = correlation_matrix(daily_ret_df)
avg_corr           = average_pairwise_correlation(corr_matrix)
sharpe_s           = sharpe_ratio(daily_ret_df)
mdd_s              = max_drawdown(nav_df)
ann_ret_s          = annualized_return(nav_df)
ann_vol_s          = annualized_volatility(daily_ret_df)
nav_snapshot_df    = get_nav_snapshot(nav_df)
health             = portfolio_health_score(sharpe_s, mdd_s, avg_corr, latest_rolling_df)
individual_health  = all_individual_scores(sharpe_s, mdd_s, latest_rolling_df)

# Benchmark risk-return metrics for scatter reference point
bench_daily_ret    = benchmark.pct_change().dropna()
_bench_ret_series  = annualized_return(benchmark.to_frame())
_bench_vol_series  = annualized_volatility(bench_daily_ret.to_frame())
benchmark_ann_ret  = float(_bench_ret_series.iloc[0]) if not _bench_ret_series.empty else 0.0
benchmark_ann_vol  = float(_bench_vol_series.iloc[0]) if not _bench_vol_series.empty else 0.0

# ---------------------------------------------------------------------------
# Render UI
# ---------------------------------------------------------------------------

# --- Section 1: Summary Card ---
render_summary_card(sharpe_s, ann_ret_s, ann_vol_s, mdd_s, fund_names)
st.divider()

# --- Section 2: NAV Snapshot ---
render_nav_snapshot(nav_snapshot_df, fund_names, fund_info)
st.divider()

# --- Section 2: Health Score ---
render_health_scorecard(health, individual_health, fund_names)
st.divider()

# --- Section 3: Key Risk Metrics ---
render_metric_cards(risk_summary_df, fund_names)
st.divider()

# --- Section 4: Rolling Returns Table ---
render_rolling_returns_table(latest_rolling_df, fund_names)
st.divider()

# --- Section 4: Benchmark Comparison ---
render_benchmark_table(excess_ret, fund_names)
st.divider()

# --- Section 5: Charts (tabbed) ---
st.subheader("📊 Charts")
render_charts_section(
    cumulative_df      = cumul_df,
    rolling_comparison = rolling_vs_bench,
    drawdown_df        = drawdown_df,
    corr_matrix        = corr_matrix,
    latest_rolling_df  = latest_rolling_df,
    ann_returns        = ann_ret_s,
    ann_volatility     = ann_vol_s,
    sharpe_series      = sharpe_s,
    benchmark_return   = benchmark_ann_ret,
    benchmark_vol      = benchmark_ann_vol,
    fund_names         = fund_names,
)

# --- Section 6: SIP Calculator ---
st.divider()
render_sip_section(nav_df, fund_names)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "📌 Data sourced from AMFI (via mftool) and NSE (via yfinance). "
    "This tool is for informational purposes only and does not constitute financial advice."
)
