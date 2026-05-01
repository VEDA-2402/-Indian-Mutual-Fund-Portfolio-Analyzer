"""
ui/components.py
----------------
Reusable Streamlit UI components: fund selector, metric cards, scorecard table, etc.
All display logic lives here — no calculation or data fetching.
"""

import streamlit as st
import pandas as pd
import numpy as np
from ui.charts import (
    plot_cumulative_returns,
    plot_rolling_returns,
    plot_rolling_returns_bar,
    plot_drawdown,
    plot_correlation_heatmap,
    plot_health_score_gauge,
    plot_score_breakdown,
    plot_sip_corpus,
    plot_risk_return_scatter,
)


# ---------------------------------------------------------------------------
# Page Config (call once in app.py)
# ---------------------------------------------------------------------------

def set_page_config(title: str, icon: str):
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ---------------------------------------------------------------------------
# Fund Selector (Sidebar) — Multi-company "Add to Portfolio" flow
# ---------------------------------------------------------------------------

def render_fund_selector(all_funds: dict) -> list:
    """
    Fund selector with persistent basket — allows adding funds from different
    companies one search at a time.

    Flow:
      1. Search any company/fund name → filtered dropdown appears
      2. Pick a fund → it gets added to the portfolio basket
      3. Clear search, type a new company → add more funds
      4. Basket persists across searches; individual funds can be removed

    Returns a list of selected scheme codes.
    """
    st.sidebar.header("🔍 Fund Selector")

    # --- Initialise session state basket ---
    if "portfolio_basket" not in st.session_state:
        st.session_state.portfolio_basket = {}  # {code: name}

    # --- Search Box ---
    search_query = st.sidebar.text_input(
        "Search fund by name or AMC",
        placeholder="e.g. Mirae, HDFC, SBI, Axis...",
        help="Search across all AMCs — add funds from different companies to compare"
    )

    # --- Filtered results ---
    if search_query.strip():
        filtered = {
            code: name
            for code, name in all_funds.items()
            if search_query.lower() in name.lower()
        }

        if not filtered:
            st.sidebar.warning("No funds found. Try a different keyword.")
        else:
            # Only show funds not already in basket
            not_added = {
                code: name for code, name in filtered.items()
                if code not in st.session_state.portfolio_basket
            }

            if not_added:
                options     = list(not_added.keys())
                format_func = lambda c: not_added.get(c, c)

                to_add = st.sidebar.selectbox(
                    f"Found {len(filtered)} fund(s) — pick one to add:",
                    options=[""] + options,
                    format_func=lambda c: "— select a fund —" if c == "" else format_func(c),
                )

                if to_add:
                    st.session_state.portfolio_basket[to_add] = all_funds.get(to_add, to_add)
                    st.rerun()
            else:
                st.sidebar.info("All matching funds already added.")

    st.sidebar.divider()

    # --- Portfolio Basket ---
    st.sidebar.markdown("#### 🗂 Your Portfolio")

    if not st.session_state.portfolio_basket:
        st.sidebar.caption("No funds added yet. Search above to add funds.")
        return []

    # Show each fund with a remove button
    to_remove = None
    for code, name in list(st.session_state.portfolio_basket.items()):
        col1, col2 = st.sidebar.columns([5, 1])
        col1.caption(f"📌 {name[:40]}")
        if col2.button("✕", key=f"remove_{code}", help="Remove this fund"):
            to_remove = code

    if to_remove:
        del st.session_state.portfolio_basket[to_remove]
        st.rerun()

    basket_codes = list(st.session_state.portfolio_basket.keys())

    st.sidebar.success(f"{len(basket_codes)} fund(s) in portfolio")

    # Clear all button
    if st.sidebar.button("🗑 Clear All Funds"):
        st.session_state.portfolio_basket = {}
        st.rerun()

    return basket_codes


# ---------------------------------------------------------------------------
# Date Range Selector (Sidebar)
# ---------------------------------------------------------------------------

def render_date_range() -> tuple:
    """
    Renders a period selector in the sidebar.
    Returns (period_label, lookback_years) for downstream use.
    """
    st.sidebar.header("📅 Analysis Period")

    period = st.sidebar.selectbox(
        "Select period",
        options=["1 Year", "3 Years", "5 Years", "Max"],
        index=2,
        help="Limits the analysis window (NAV data will be sliced accordingly)"
    )

    period_map = {
        "1 Year":  1,
        "3 Years": 3,
        "5 Years": 5,
        "Max":     None,
    }
    return period, period_map[period]


# ---------------------------------------------------------------------------
# Metric Cards Row
# ---------------------------------------------------------------------------

def render_metric_cards(risk_summary_df: pd.DataFrame, fund_names: dict = None):
    """
    Renders a row of metric cards showing key stats per fund.
    risk_summary_df: DataFrame with cols [Volatility, Sharpe Ratio, Max Drawdown], index = scheme codes
    """
    st.subheader("📊 Key Risk Metrics")

    for code in risk_summary_df.index:
        label    = fund_names.get(code, code) if fund_names else code
        vol      = risk_summary_df.loc[code, "Volatility (Ann.)"]
        sharpe   = risk_summary_df.loc[code, "Sharpe Ratio"]
        mdd      = risk_summary_df.loc[code, "Max Drawdown"]

        with st.expander(f"📁 {label[:60]}", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Annualized Volatility",
                f"{vol * 100:.1f}%",
                help="Standard deviation of daily returns × √252"
            )
            c2.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                delta="Good" if sharpe >= 1.0 else "Review",
                delta_color="normal" if sharpe >= 1.0 else "inverse",
                help="Risk-adjusted return (>1.0 is good for Indian MFs)"
            )
            c3.metric(
                "Max Drawdown",
                f"{mdd * 100:.1f}%",
                help="Worst peak-to-trough decline in the selected period"
            )


# ---------------------------------------------------------------------------
# Fund Comparison Summary Card
# ---------------------------------------------------------------------------

def render_summary_card(
    sharpe_series: pd.Series,
    ann_returns: pd.Series,
    ann_volatility: pd.Series,
    max_drawdown_series: pd.Series,
    fund_names: dict = None,
):
    """
    Renders a single "winner" recommendation card at the very top.
    Picks the best fund on 3 dimensions and calls out the weakest too.
    """
    if sharpe_series.empty or ann_returns.empty:
        return

    # --- Find winners per dimension ---
    best_sharpe_code   = sharpe_series.idxmax()
    best_return_code   = ann_returns.idxmax()
    best_low_risk_code = ann_volatility.idxmin()
    worst_dd_code      = max_drawdown_series.idxmin()   # most negative = worst drawdown
    weakest_code       = sharpe_series.idxmin()

    def name(code):
        n = fund_names.get(code, code) if fund_names else code
        return n[:45]

    best_sharpe_val = sharpe_series[best_sharpe_code]
    best_ret_val    = ann_returns[best_return_code] * 100
    best_vol_val    = ann_volatility[best_low_risk_code] * 100
    worst_dd_val    = max_drawdown_series[worst_dd_code] * 100
    weakest_sharpe  = sharpe_series[weakest_code]

    # --- Overall winner = highest Sharpe (best risk-adjusted) ---
    overall_winner = best_sharpe_code

    st.markdown("### 🏆 Portfolio Summary")

    # Top winner card
    winner_col, stats_col = st.columns([2, 3])

    with winner_col:
        with st.container(border=True):
            st.markdown(
                f"#### 🥇 Strongest Fund\n"
                f"**{name(overall_winner)}**\n\n"
                f"Best risk-adjusted return in your portfolio with a Sharpe Ratio of **{best_sharpe_val:.2f}**."
            )

    with stats_col:
        with st.container(border=True):
            st.markdown("#### 📊 Category Leaders")
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Best Returns",
                name(best_return_code)[:22],
                delta=f"{best_ret_val:.1f}% p.a.",
                help="Highest annualised return"
            )
            c2.metric(
                "Lowest Risk",
                name(best_low_risk_code)[:22],
                delta=f"{best_vol_val:.1f}% vol",
                delta_color="inverse",
                help="Lowest annualised volatility"
            )
            c3.metric(
                "Worst Drawdown",
                name(worst_dd_code)[:22],
                delta=f"{worst_dd_val:.1f}%",
                delta_color="inverse",
                help="Fund with the largest peak-to-trough decline"
            )

    # Weak fund callout (only if more than 1 fund)
    if len(sharpe_series) > 1 and weakest_code != overall_winner:
        st.warning(
            f"⚠️ **Review:** {name(weakest_code)} has the lowest Sharpe Ratio ({weakest_sharpe:.2f}) "
            f"in your portfolio — its returns may not justify the risk it carries."
        )


# ---------------------------------------------------------------------------
# NAV Snapshot — Today's NAV + 52-week High/Low
# ---------------------------------------------------------------------------

def render_nav_snapshot(nav_snapshot_df: pd.DataFrame, fund_names: dict = None, fund_info: dict = None):
    """
    Renders a compact snapshot card per fund showing:
    - Fund category / type badge
    - Current NAV + last updated date
    - 52-week High (and % below it)
    - 52-week Low (and % above it)
    - A small progress bar showing where NAV sits in the 52w range
    """
    st.subheader("📍 NAV Snapshot")
    st.caption("Current NAV and 52-week price range for each fund.")

    for code, row in nav_snapshot_df.iterrows():
        name = fund_names.get(code, code)[:60] if fund_names else code

        # Fund category info
        info      = (fund_info or {}).get(code, {})
        category  = info.get("category", "N/A")
        fund_type = info.get("fund_type", "N/A")
        fund_house = info.get("fund_house", "")

        # Position in 52w range (0% = at low, 100% = at high)
        rng = row["week52_high"] - row["week52_low"]
        pos_in_range = ((row["latest_nav"] - row["week52_low"]) / rng * 100) if rng > 0 else 50.0
        pos_in_range = max(0.0, min(100.0, pos_in_range))

        # Colour for distance from high
        pct_high = row["pct_from_high"]
        if pct_high >= -5:
            high_color = "#2ecc71"
        elif pct_high >= -15:
            high_color = "#e67e22"
        else:
            high_color = "#e74c3c"

        with st.container(border=True):
            # Header row: name + category badges
            head_col, badge_col = st.columns([3, 2])
            with head_col:
                st.markdown(f"**📌 {name}**")
            with badge_col:
                badges = []
                if category != "N/A":
                    badges.append(f"`{category}`")
                if fund_type != "N/A":
                    badges.append(f"`{fund_type}`")
                if badges:
                    st.markdown("&nbsp;" + "  ".join(badges), unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                label="Current NAV",
                value=f"₹{row['latest_nav']:,.4f}",
                help=f"As of {row['latest_date']}" + (f" · {fund_house}" if fund_house else "")
            )
            c2.metric(
                label="52W High",
                value=f"₹{row['week52_high']:,.4f}",
                delta=f"{pct_high:.1f}% from peak",
                delta_color="inverse",   # red when negative (below peak)
                help=f"Reached on {row['week52_high_date']}"
            )
            c3.metric(
                label="52W Low",
                value=f"₹{row['week52_low']:,.4f}",
                delta=f"+{row['pct_from_low']:.1f}% above trough",
                delta_color="normal",
                help=f"Reached on {row['week52_low_date']}"
            )

            # Progress bar showing position in 52w range
            with c4:
                st.caption("Position in 52W Range")
                st.progress(
                    value=int(pos_in_range),
                    text=f"{pos_in_range:.0f}% of range"
                )


# ---------------------------------------------------------------------------
# Rolling Returns Table
# ---------------------------------------------------------------------------

def render_rolling_returns_table(latest_rolling_df: pd.DataFrame, fund_names: dict = None):
    """
    Renders a color-coded table of latest rolling returns.
    latest_rolling_df: rows = windows, cols = scheme codes
    """
    st.subheader("📈 Latest Rolling Returns")

    display_df = latest_rolling_df.copy() * 100  # convert to percentage

    # Rename columns to fund names if available
    if fund_names:
        display_df.columns = [
            fund_names.get(c, c)[:35] for c in display_df.columns
        ]

    display_df = display_df.round(2)

    # Style: green = positive, red = negative
    def color_cells(val):
        if pd.isna(val):
            return "color: gray"
        color = "#2ecc71" if val > 0 else "#e74c3c"
        return f"color: {color}; font-weight: bold"

    styled = display_df.style.map(color_cells).format("{:.2f}%", na_rep="N/A")
    st.dataframe(styled, use_container_width=True)


# ---------------------------------------------------------------------------
# Benchmark Comparison Table
# ---------------------------------------------------------------------------

def render_benchmark_table(excess_returns: pd.Series, fund_names: dict = None):
    """
    Renders a table showing annualized excess return of each fund vs Nifty 50.
    excess_returns: pd.Series indexed by scheme code
    """
    st.subheader("📊 Performance vs Nifty 50 Benchmark")

    df = excess_returns.rename("Excess Return (Ann.)").to_frame()
    df["vs Benchmark"] = df["Excess Return (Ann.)"].apply(
        lambda x: "✅ Outperforming" if x > 0 else "❌ Underperforming"
    )
    df["Excess Return (Ann.)"] = (df["Excess Return (Ann.)"] * 100).round(2).astype(str) + "%"

    if fund_names:
        df.index = [fund_names.get(c, c)[:40] for c in df.index]

    st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# Portfolio Health Scorecard
# ---------------------------------------------------------------------------

def _render_single_scorecard(score_result: dict, title: str = None):
    """
    Internal helper: renders one gauge + breakdown + interpretation block.
    Used by both combined and individual score views.
    """
    if title:
        st.markdown(f"##### {title}")

    grade = score_result["grade"]
    score = score_result["total_score"]

    # Grade colours
    grade_colors = {"A": "#2ecc71", "B": "#f1c40f", "C": "#e67e22", "D": "#e74c3c", "F": "#c0392b"}
    grade_color  = grade_colors.get(grade, "#95a5a6")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.plotly_chart(
            plot_health_score_gauge(score, grade),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            plot_score_breakdown(score_result["components"]),
            use_container_width=True
        )

    # Breakdown table
    weight_key_map = {
        "Sharpe Ratio":       "sharpe",
        "Max Drawdown":       "drawdown",
        "Diversification":    "correlation",
        "Consistent Returns": "returns",
    }
    rows = []
    for name, data in score_result["components"].items():
        key = weight_key_map.get(name, "")
        rows.append({
            "Category":    name,
            "Description": data["label"],
            "Raw Value":   data["raw_value"] if data["raw_value"] is not None else "N/A",
            "Score":       f"{data['score']:.0f} / 100",
            "Weight":      f"{score_result['weights'].get(key, 0) * 100:.0f}%",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Category"), use_container_width=True)
    st.info(f"💬 **Insight:** {score_result['interpretation']}")


def render_health_scorecard(portfolio_score: dict, individual_scores: list, fund_names: dict = None):
    """
    Renders Portfolio Health Score section with a toggle:
      - Combined: one score for the whole portfolio
      - Individual: one gauge per fund side by side
    """
    st.subheader("🏥 Portfolio Health Score")

    view = st.radio(
        "View",
        options=["🏦 Combined Portfolio Score", "📊 Individual Fund Scores"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()

    if view == "🏦 Combined Portfolio Score":
        _render_single_scorecard(portfolio_score, title="Combined Portfolio Score")

    else:
        # Individual scores — laid out in columns (2 per row)
        if not individual_scores:
            st.info("No individual scores available.")
            return

        for i in range(0, len(individual_scores), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j >= len(individual_scores):
                    break
                result = individual_scores[i + j]
                code   = result.get("scheme_code", "")
                name   = fund_names.get(code, code)[:50] if fund_names else code
                with col:
                    with st.container(border=True):
                        _render_single_scorecard(result, title=f"📌 {name}")


# ---------------------------------------------------------------------------
# Charts Section
# ---------------------------------------------------------------------------

def render_charts_section(
    cumulative_df: pd.DataFrame,
    rolling_comparison: dict,
    drawdown_df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    latest_rolling_df: pd.DataFrame,
    ann_returns: pd.Series = None,
    ann_volatility: pd.Series = None,
    sharpe_series: pd.Series = None,
    benchmark_return: float = None,
    benchmark_vol: float = None,
    fund_names: dict = None,
):
    """
    Renders all chart sections with tabs for organization.
    """
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Returns", "📉 Drawdown", "🔗 Correlation", "🔄 Rolling Returns", "🎯 Risk-Return"
    ])

    with tab1:
        st.plotly_chart(
            plot_cumulative_returns(cumulative_df, fund_names),
            use_container_width=True
        )
        st.plotly_chart(
            plot_rolling_returns_bar(latest_rolling_df, fund_names),
            use_container_width=True
        )

    with tab2:
        st.plotly_chart(
            plot_drawdown(drawdown_df, fund_names),
            use_container_width=True
        )

    with tab3:
        if len(corr_matrix) >= 2:
            st.plotly_chart(
                plot_correlation_heatmap(corr_matrix, fund_names),
                use_container_width=True
            )
        else:
            st.info("Select at least 2 funds to see the correlation matrix.")

    with tab4:
        window = st.selectbox(
            "Select Rolling Window",
            options=list(rolling_comparison.keys()),
            index=min(3, len(rolling_comparison) - 1),
        )
        if window in rolling_comparison and not rolling_comparison[window].empty:
            st.plotly_chart(
                plot_rolling_returns(rolling_comparison[window], window, fund_names),
                use_container_width=True
            )
        else:
            st.warning(f"Not enough data to show {window} rolling returns.")

    with tab5:
        if ann_returns is not None and ann_volatility is not None and sharpe_series is not None:
            st.caption(
                "Each bubble is one fund. **X = Risk (volatility)**, **Y = Return**. "
                "Bigger bubble = higher Sharpe ratio. "
                "Top-left quadrant = best (high return, low risk)."
            )
            st.plotly_chart(
                plot_risk_return_scatter(
                    ann_returns, ann_volatility, sharpe_series,
                    benchmark_return, benchmark_vol, fund_names
                ),
                use_container_width=True
            )
        else:
            st.info("Risk-Return data not available.")


# ---------------------------------------------------------------------------
# SIP Calculator Section
# ---------------------------------------------------------------------------

def render_sip_section(nav_df: pd.DataFrame, fund_names: dict = None):
    """
    Renders the full SIP Calculator section:
    - Monthly amount input
    - Summary cards (total invested, final value, gain, best fund)
    - XIRR per fund table
    - Invested vs Corpus chart
    """
    from calculations.returns import sip_analysis

    st.subheader("💰 SIP Calculator")
    st.caption("Simulate a monthly SIP using actual historical NAVs for your selected funds.")

    col_input, col_note = st.columns([1, 2])
    with col_input:
        monthly_amount = st.number_input(
            "Monthly SIP Amount (₹) per fund",
            min_value=500,
            max_value=1_000_000,
            value=5000,
            step=500,
            help="This amount is invested in EACH selected fund every month"
        )

    with col_note:
        st.info(
            f"💡 ₹{monthly_amount:,}/month per fund × {len(nav_df.columns)} fund(s) "
            f"= ₹{monthly_amount * len(nav_df.columns):,}/month total SIP"
        )

    # Persist SIP results in session state so they survive reruns
    if "sip_result" not in st.session_state:
        st.session_state.sip_result       = None
        st.session_state.sip_amount_used  = None
        st.session_state.sip_funds_used   = None

    run_col, clear_col = st.columns([2, 1])
    with run_col:
        run = st.button("▶ Run SIP Simulation", type="primary")
    with clear_col:
        if st.button("✕ Clear Results") and st.session_state.sip_result is not None:
            st.session_state.sip_result      = None
            st.session_state.sip_amount_used = None
            st.session_state.sip_funds_used  = None
            st.rerun()

    # Detect if funds or amount changed since last run — prompt re-run
    current_funds = sorted(nav_df.columns.tolist())
    if (
        st.session_state.sip_result is not None and
        (st.session_state.sip_amount_used != monthly_amount or
         st.session_state.sip_funds_used  != current_funds)
    ):
        st.warning("⚠️ Portfolio or amount changed — click **▶ Run SIP Simulation** to refresh results.")

    if run:
        with st.spinner("Simulating SIP..."):
            try:
                result = sip_analysis(nav_df, monthly_amount)
                st.session_state.sip_result      = result
                st.session_state.sip_amount_used = monthly_amount
                st.session_state.sip_funds_used  = current_funds
            except RuntimeError as e:
                st.error(f"⚠️ {e}")
                return

    # Render persisted result if available
    if st.session_state.sip_result is None:
        return

    result  = st.session_state.sip_result
    summary = result["summary"]
    best    = result["best_fund"]

        # --- Summary cards ---
    st.markdown("#### 📋 Summary")
    total_invested = summary["total_invested"].sum()
    total_value    = summary["final_value"].sum()
    total_gain     = summary["gain"].sum()
    total_gain_pct = (total_gain / total_invested * 100) if total_invested > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Invested", f"₹{total_invested:,.0f}")
    c2.metric("Current Value", f"₹{total_value:,.0f}",
              delta=f"₹{total_gain:,.0f} ({total_gain_pct:.1f}%)",
              delta_color="normal" if total_gain >= 0 else "inverse")
    c3.metric("Total Gain", f"₹{total_gain:,.0f}")
    if best:
        best_name = fund_names.get(best, best)[:30] if fund_names else best
        best_xirr = summary.loc[best, "xirr"]
        c4.metric("Best Fund (XIRR)", best_name, delta=f"{best_xirr*100:.1f}% p.a.")

    # --- XIRR per fund table ---
    st.markdown("#### 📊 Per-Fund XIRR Breakdown")
    table_df = summary[["total_invested", "final_value", "gain", "gain_pct", "xirr", "sip_count"]].copy()
    table_df.index = [fund_names.get(c, c)[:45] if fund_names else c for c in table_df.index]
    table_df.columns = ["Invested (₹)", "Final Value (₹)", "Gain (₹)", "Gain (%)", "XIRR (p.a.)", "SIP Instalments"]
    table_df["Invested (₹)"]    = table_df["Invested (₹)"].map("₹{:,.0f}".format)
    table_df["Final Value (₹)"] = table_df["Final Value (₹)"].map("₹{:,.0f}".format)
    table_df["Gain (₹)"]        = table_df["Gain (₹)"].map("₹{:,.0f}".format)
    table_df["Gain (%)"]        = table_df["Gain (%)"].map("{:.1f}%".format)
    table_df["XIRR (p.a.)"]     = table_df["XIRR (p.a.)"].map(
        lambda x: f"{x*100:.1f}%" if not pd.isna(x) else "N/A"
    )
    table_df["SIP Instalments"] = table_df["SIP Instalments"].astype(int)
    st.dataframe(table_df, use_container_width=True)

    # --- Corpus vs Invested chart ---
    st.markdown("#### 📈 Portfolio Value vs Amount Invested")
    st.plotly_chart(
        plot_sip_corpus(result["corpus_df"], result["invested_df"], fund_names),
        use_container_width=True
    )


# ---------------------------------------------------------------------------
# Error / Loading States
# ---------------------------------------------------------------------------

def render_error(message: str):
    st.error(f"⚠️ {message}")


def render_loading_placeholder():
    # -----------------------------------------------------------------------
    # Hero Section
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 1rem 1rem 1rem;">
            <h1 style="font-size:2.6rem; font-weight:800; margin-bottom:0.3rem;">
                📈 Indian MF Portfolio Analyzer
            </h1>
            <p style="font-size:1.15rem; color:#6C757D; max-width:620px; margin:0 auto 1.2rem auto;">
                Professional-grade portfolio analysis for Indian mutual funds —
                returns, risk, correlation, health score, and AI-powered recommendations.
                All powered by real AMFI data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Live Stats Bar
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <div style="
            display:flex; justify-content:center; gap:2rem;
            background: linear-gradient(135deg, #0D1B2A 0%, #1A3550 100%);
            border-radius:12px; padding:1.2rem 2rem; margin:0.5rem 0 1.5rem 0;
            flex-wrap:wrap;
        ">
            <div style="text-align:center; color:white;">
                <div style="font-size:1.8rem; font-weight:800; color:#4FC3F7;">10,000+</div>
                <div style="font-size:0.8rem; color:#90CAF9;">Mutual Funds Tracked</div>
            </div>
            <div style="text-align:center; color:white;">
                <div style="font-size:1.8rem; font-weight:800; color:#81C784;">Live</div>
                <div style="font-size:0.8rem; color:#90CAF9;">AMFI NAV Data</div>
            </div>
            <div style="text-align:center; color:white;">
                <div style="font-size:1.8rem; font-weight:800; color:#FFB74D;">6</div>
                <div style="font-size:0.8rem; color:#90CAF9;">Rolling Return Windows</div>
            </div>
            <div style="text-align:center; color:white;">
                <div style="font-size:1.8rem; font-weight:800; color:#F48FB1;">100</div>
                <div style="font-size:0.8rem; color:#90CAF9;">Portfolio Health Score</div>
            </div>
            <div style="text-align:center; color:white;">
                <div style="font-size:1.8rem; font-weight:800; color:#CE93D8;">3</div>
                <div style="font-size:0.8rem; color:#90CAF9;">AI Portfolio Options</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Quick Start — pre-loads 3 famous funds on click
    # -----------------------------------------------------------------------
    QUICK_START_FUNDS = {
        "118778": "Parag Parikh Flexi Cap Fund - Regular Plan - Growth",
        "120503": "Mirae Asset Large Cap Fund - Regular Plan - Growth",
        "125497": "Nippon India Small Cap Fund - Regular Plan - Growth",
    }

    st.markdown(
        """
        <div style="text-align:center; margin-bottom:0.5rem;">
            <p style="color:#6C757D; font-size:0.95rem;">
                Not sure where to start? Load a sample portfolio instantly:
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    qs_col1, qs_col2, qs_col3 = st.columns([1, 2, 1])
    with qs_col2:
        if st.button(
            "⚡ Quick Start — Load Sample Portfolio",
            use_container_width=True,
            type="primary",
            help="Loads Parag Parikh Flexi Cap + Mirae Asset Large Cap + Nippon Small Cap",
        ):
            if "portfolio_basket" not in st.session_state:
                st.session_state.portfolio_basket = {}
            st.session_state.portfolio_basket.update(QUICK_START_FUNDS)
            st.rerun()

    st.markdown(
        "<p style='text-align:center; color:#9E9E9E; font-size:0.8rem; margin-top:0.3rem;'>"
        "Parag Parikh Flexi Cap &nbsp;•&nbsp; Mirae Asset Large Cap &nbsp;•&nbsp; Nippon Small Cap"
        "</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Feature Cards
    # -----------------------------------------------------------------------
    st.markdown(
        "<h3 style='text-align:center; margin-bottom:1rem;'>What you get</h3>",
        unsafe_allow_html=True,
    )

    f1, f2, f3, f4 = st.columns(4)

    f1.markdown(
        """
        <div style="background:#000000; border-radius:10px; padding:1rem; text-align:center; height:140px;">
            <div style="font-size:2rem;">📊</div>
            <div style="font-weight:700; margin:0.3rem 0 0.2rem; color:#ffffff;">Rolling Returns</div>
            <div style="font-size:0.8rem; color:#cccccc;">1M · 3M · 6M · 1Y · 3Y · 5Y windows with benchmark overlay</div>
        </div>
        """, unsafe_allow_html=True,
    )
    f2.markdown(
        """
        <div style="background:#000000; border-radius:10px; padding:1rem; text-align:center; height:140px;">
            <div style="font-size:2rem;">⚡</div>
            <div style="font-weight:700; margin:0.3rem 0 0.2rem; color:#ffffff;">Risk Analysis</div>
            <div style="font-size:0.8rem; color:#cccccc;">Sharpe ratio, max drawdown, volatility and correlation matrix</div>
        </div>
        """, unsafe_allow_html=True,
    )
    f3.markdown(
        """
        <div style="background:#000000; border-radius:10px; padding:1rem; text-align:center; height:140px;">
            <div style="font-size:2rem;">🤖</div>
            <div style="font-weight:700; margin:0.3rem 0 0.2rem; color:#ffffff;">AI Recommender</div>
            <div style="font-size:0.8rem; color:#cccccc;">3 diversified portfolio options based on your risk profile</div>
        </div>
        """, unsafe_allow_html=True,
    )
    f4.markdown(
        """
        <div style="background:#000000; border-radius:10px; padding:1rem; text-align:center; height:140px;">
            <div style="font-size:2rem;">📄</div>
            <div style="font-weight:700; margin:0.3rem 0 0.2rem; color:#ffffff;">PDF Export</div>
            <div style="font-size:0.8rem; color:#cccccc;">Full analyst-style report with diagnosis, tables, and health score</div>
        </div>
        """, unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # How to use steps
    # -----------------------------------------------------------------------
    st.markdown(
        "<h3 style='text-align:center; margin-bottom:1rem;'>How to use</h3>",
        unsafe_allow_html=True,
    )

    s1, s2, s3, s4 = st.columns(4)
    for col, num, title, desc in [
        (s1, "1", "Search", "Type any AMC or fund name in the sidebar search box"),
        (s2, "2", "Add Funds", "Add 2-5 funds from different categories to your portfolio"),
        (s3, "3", "Analyze", "Instantly see returns, risk metrics, charts, and health score"),
        (s4, "4", "Export", "Download a full PDF report or get AI fund recommendations"),
    ]:
        col.markdown(
            f"""
            <div style="text-align:center; padding:0.8rem;">
                <div style="
                    width:36px; height:36px; border-radius:50%;
                    background:#0D1B2A; color:white;
                    font-size:1.1rem; font-weight:800;
                    display:flex; align-items:center; justify-content:center;
                    margin:0 auto 0.5rem auto;
                ">{num}</div>
                <div style="font-weight:700; margin-bottom:0.2rem;">{title}</div>
                <div style="font-size:0.8rem; color:#6C757D;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption(
        "Data sourced from AMFI (via mftool) and NSE (via yfinance). "
        "For informational purposes only — not financial advice."
    )


# ---------------------------------------------------------------------------
# Fund Recommender Tab
# ---------------------------------------------------------------------------

def render_recommender_tab():
    """
    Renders the Fund Recommender tab.
    Displays 3 model portfolio options (each with 3 diversified funds),
    blended SIP projection, and plain-English reasoning per option.
    """
    from recommender import get_portfolio_recommendations, horizon_to_years

    st.subheader("🤖 Fund Recommender")
    st.caption(
        "Tell us your risk appetite and horizon — we'll suggest **3 model portfolios**, "
        "each with 3 diversified funds. Pick the one that matches your gut."
    )
    st.divider()

    # --- User Inputs ---
    col1, col2, col3 = st.columns(3)

    with col1:
        risk = st.selectbox(
            "🎯 Risk Appetite",
            options=["Low", "Medium", "High"],
            index=1,
            help="Low → Debt/Hybrid | Medium → Large Cap/Flexi/ELSS | High → Mid/Small Cap",
        )
    with col2:
        horizon = st.selectbox(
            "⏳ Investment Horizon",
            options=["Short (1Y)", "Medium (3Y)", "Long (5Y+)"],
            index=1,
            help="How long you plan to stay invested.",
        )
    with col3:
        monthly_sip = st.number_input(
            "💰 Monthly SIP Amount (₹)",
            min_value=500,
            max_value=500000,
            value=5000,
            step=500,
            help="Total monthly SIP — split equally across 3 funds in each portfolio.",
        )

    st.divider()

    # --- Profile summary ---
    risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "🟡")
    years = horizon_to_years(horizon)
    st.markdown(
        f"**Your Profile:** {risk_color} {risk} Risk &nbsp;|&nbsp; "
        f"⏳ {horizon} &nbsp;|&nbsp; "
        f"💰 ₹{monthly_sip:,}/month &nbsp;|&nbsp; "
        f"₹{monthly_sip // 3:,}/fund/month (split equally)"
    )

    recommend_btn = st.button(
        "🔍 Show My 3 Portfolio Options", type="primary", use_container_width=True
    )

    if not recommend_btn:
        st.markdown("####")
        i1, i2, i3 = st.columns(3)
        i1.info("**Step 1**\n\nSet your risk level, horizon and SIP amount above.")
        i2.info("**Step 2**\n\nClick the button — we'll build 3 diversified portfolio options.")
        i3.info("**Step 3**\n\nCompare the options and pick the one that suits your thinking.")
        return

    # --- Run engine ---
    progress_bar = st.progress(0, text="Analyzing fund universe...")
    status_text  = st.empty()

    def on_progress(current, total, fund_name):
        pct = int((current / total) * 100)
        progress_bar.progress(pct, text=f"Analyzing {current}/{total}: {fund_name[:50]}...")
        status_text.caption(f"Fetching data for: **{fund_name}**")

    try:
        portfolios = get_portfolio_recommendations(
            risk=risk,
            horizon=horizon,
            monthly_sip=float(monthly_sip),
            progress_callback=on_progress,
        )
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(
            f"❌ Could not fetch fund data. Please check your internet connection.\n\nDetails: {e}"
        )
        return

    progress_bar.empty()
    status_text.empty()

    if not portfolios:
        st.warning("⚠️ No portfolios could be built. Try a different risk level or horizon.")
        return

    st.success(f"✅ Here are your 3 portfolio options for {risk} risk, {horizon} horizon!")
    st.divider()

    fund_rank_emoji = ["🥇", "🥈", "🥉"]

    for portfolio in portfolios:
        sip    = portfolio["sip"]
        funds  = portfolio["funds"]

        with st.container(border=True):

            # --- Portfolio header ---
            ph1, ph2 = st.columns([5, 1])
            with ph1:
                st.markdown(f"### {portfolio['name']}")
                st.caption(portfolio["summary"])
            with ph2:
                st.markdown(
                    f"<div style='text-align:center; padding:10px 6px; border-radius:8px; background:#000000;'>"
                    f"<b style='font-size:13px; color:#ffffff;'>₹{sip['total_projected']:,.0f}</b><br>"
                    f"<span style='font-size:10px; color:#cccccc;'>Projected ({years}Y)</span></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # --- 3 fund rows ---
            for idx, fund in enumerate(funds):
                m = fund["metrics"]
                ann_ret = m.get("ann_return", np.nan)
                sharpe  = m.get("sharpe", np.nan)
                mdd     = m.get("max_drawdown", np.nan)
                vol     = m.get("ann_volatility", np.nan)
                emoji   = fund_rank_emoji[idx] if idx < len(fund_rank_emoji) else "•"

                fc1, fc2, fc3, fc4, fc5 = st.columns([3, 1, 1, 1, 1])

                with fc1:
                    st.markdown(f"**{emoji} {fund['fund_name']}**")
                    st.caption(f"{fund['category']} · `{fund['scheme_code']}` · {fund['reason']}")
                fc2.metric("Ann. Return",  f"{ann_ret*100:.1f}%" if not np.isnan(ann_ret) else "N/A")
                fc3.metric("Sharpe",       f"{sharpe:.2f}"        if not np.isnan(sharpe)  else "N/A")
                fc4.metric("Max Drawdown", f"{mdd*100:.1f}%"      if not np.isnan(mdd)     else "N/A")
                fc5.metric("Volatility",   f"{vol*100:.1f}%"      if not np.isnan(vol)     else "N/A")

                if idx < len(funds) - 1:
                    st.markdown("")   # small spacer between funds

            st.markdown("---")

            # --- Blended SIP summary ---
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("💰 Per Fund SIP",    f"₹{sip['per_fund_sip']:,.0f}/mo",
                      help="Your monthly SIP split equally across the 3 funds.")
            s2.metric("📦 Total Invested",  f"₹{sip['total_invested']:,.0f}",
                      help=f"Total amount invested over {years} years.")
            s3.metric(f"🚀 Projected Corpus ({years}Y)", f"₹{sip['total_projected']:,.0f}",
                      help="Estimated corpus based on each fund's historical annual return.")
            s4.metric("📊 Total Gain",      f"₹{sip['total_gain']:,.0f}",
                      delta=f"+{sip['gain_pct']:.1f}% over invested",
                      help="Gain = Projected corpus − Total invested.")

        st.markdown("####")

    # --- Disclaimer ---
    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** Recommendations are based on historical performance metrics only. "
        "Past performance is not a guarantee of future returns. "
        "This tool is for educational purposes and does not constitute financial advice. "
        "Please consult a SEBI-registered financial advisor before investing."
    )
