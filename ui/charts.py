"""
ui/charts.py
------------
All Plotly chart builders for the Streamlit app.
Each function accepts pre-computed DataFrames and returns a plotly Figure.
No data fetching or calculation logic here — pure visualization.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------

COLORS = px.colors.qualitative.Set2  # colorblind-friendly


def _fund_color_map(columns: list) -> dict:
    return {col: COLORS[i % len(COLORS)] for i, col in enumerate(columns)}


# ---------------------------------------------------------------------------
# 1. Cumulative Returns Line Chart
# ---------------------------------------------------------------------------

def plot_cumulative_returns(cumulative_df: pd.DataFrame, fund_names: dict = None) -> go.Figure:
    """
    Line chart of cumulative returns over time for all selected funds.
    cumulative_df: DataFrame with DatetimeIndex, columns = scheme codes, values = fraction (e.g. 0.15 = 15%)
    fund_names: optional dict {scheme_code: fund_name} for display labels
    """
    fig = go.Figure()
    color_map = _fund_color_map(cumulative_df.columns.tolist())

    for col in cumulative_df.columns:
        label = fund_names.get(col, col) if fund_names else col
        fig.add_trace(go.Scatter(
            x=cumulative_df.index,
            y=cumulative_df[col] * 100,
            mode="lines",
            name=label,
            line=dict(color=color_map[col], width=2),
            hovertemplate="%{x|%b %Y}<br>Return: %{y:.1f}%<extra>" + label + "</extra>"
        ))

    fig.update_layout(
        title="Cumulative Returns Over Time",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=420,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


# ---------------------------------------------------------------------------
# 2. Rolling Returns Line Chart
# ---------------------------------------------------------------------------

def plot_rolling_returns(rolling_df: pd.DataFrame, window_label: str, fund_names: dict = None) -> go.Figure:
    """
    Line chart of rolling returns for a specific window (e.g. '1Y').
    rolling_df: DataFrame with DatetimeIndex, columns = scheme codes + optionally 'Nifty50'
    """
    fig = go.Figure()
    color_map = _fund_color_map(rolling_df.columns.tolist())

    for col in rolling_df.columns:
        is_benchmark = col == "Nifty50"
        label = fund_names.get(col, col) if (fund_names and not is_benchmark) else col
        fig.add_trace(go.Scatter(
            x=rolling_df.index,
            y=rolling_df[col] * 100,
            mode="lines",
            name=label,
            line=dict(
                color=color_map[col],
                width=2,
                dash="dot" if is_benchmark else "solid"
            ),
            hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<extra>" + label + "</extra>"
        ))

    fig.update_layout(
        title=f"{window_label} Rolling Returns",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=420,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


# ---------------------------------------------------------------------------
# 3. Drawdown Chart
# ---------------------------------------------------------------------------

def plot_drawdown(drawdown_df: pd.DataFrame, fund_names: dict = None) -> go.Figure:
    """
    Area chart showing drawdown over time for each fund.
    drawdown_df: DataFrame with DatetimeIndex, columns = scheme codes, values = negative fractions
    """
    fig = go.Figure()
    color_map = _fund_color_map(drawdown_df.columns.tolist())

    for col in drawdown_df.columns:
        label = fund_names.get(col, col) if fund_names else col
        fig.add_trace(go.Scatter(
            x=drawdown_df.index,
            y=drawdown_df[col] * 100,
            mode="lines",
            fill="tozeroy",
            name=label,
            line=dict(color=color_map[col], width=1.5),
            fillcolor=color_map[col].replace("rgb", "rgba").replace(")", ", 0.15)") if "rgb" in color_map[col] else color_map[col],
            hovertemplate="%{x|%b %Y}<br>Drawdown: %{y:.1f}%<extra>" + label + "</extra>"
        ))

    fig.update_layout(
        title="Drawdown Over Time",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=400,
        yaxis=dict(tickformat=".0f"),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(corr_matrix: pd.DataFrame, fund_names: dict = None) -> go.Figure:
    """
    Heatmap of pairwise fund correlations.
    corr_matrix: square DataFrame (scheme codes x scheme codes)
    """
    full_labels = [fund_names.get(c, c) if fund_names else c for c in corr_matrix.columns]

    # Build smart short labels: take key words to stay under 18 chars
    def _shorten(name: str, max_len: int = 18) -> str:
        if len(name) <= max_len:
            return name
        # Try: first word + last word
        parts = name.split()
        if len(parts) >= 2:
            candidate = f"{parts[0]} {parts[-1]}"
            if len(candidate) <= max_len:
                return candidate
        return name[:max_len - 1] + "…"

    short_labels = [_shorten(l) for l in full_labels]

    # Hover uses full names
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=short_labels,
        y=short_labels,
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=11),
        customdata=[[f"{a} vs {b}" for b in full_labels] for a in full_labels],
        hovertemplate="%{customdata}<br>Correlation: %{z:.2f}<extra></extra>",
        colorbar=dict(title="Corr.")
    ))

    fig.update_layout(
        title="Fund Correlation Matrix",
        template="plotly_white",
        height=max(380, 90 * len(corr_matrix)),
        xaxis=dict(tickangle=-30, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
        margin=dict(l=140, b=120),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Rolling Returns Bar Chart (Latest Snapshot)
# ---------------------------------------------------------------------------

def plot_rolling_returns_bar(latest_rolling_df: pd.DataFrame, fund_names: dict = None) -> go.Figure:
    """
    Grouped bar chart showing latest rolling returns across all windows for each fund.
    latest_rolling_df: DataFrame — rows = windows (1M, 3M...), cols = scheme codes
    """
    fig = go.Figure()
    color_map = _fund_color_map(latest_rolling_df.columns.tolist())

    for col in latest_rolling_df.columns:
        label = fund_names.get(col, col) if fund_names else col
        fig.add_trace(go.Bar(
            name=label,
            x=latest_rolling_df.index.tolist(),
            y=(latest_rolling_df[col] * 100).round(2).tolist(),
            marker_color=color_map[col],
            hovertemplate="%{x}: %{y:.1f}%<extra>" + label + "</extra>"
        ))

    fig.update_layout(
        title="Latest Rolling Returns by Window",
        xaxis_title="Window",
        yaxis_title="Return (%)",
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    return fig


# ---------------------------------------------------------------------------
# 6. Portfolio Health Score Gauge
# ---------------------------------------------------------------------------

def plot_health_score_gauge(score: float, grade: str) -> go.Figure:
    """
    Gauge chart for the Portfolio Health Score (0–100).
    """
    if score >= 80:
        bar_color = "#2ecc71"   # green
    elif score >= 65:
        bar_color = "#f1c40f"   # yellow
    elif score >= 50:
        bar_color = "#e67e22"   # orange
    else:
        bar_color = "#e74c3c"   # red

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": f"  ({grade})", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": bar_color, "thickness": 0.3},
            "steps": [
                {"range": [0,  35], "color": "#fdecea"},
                {"range": [35, 50], "color": "#fef9e7"},
                {"range": [50, 65], "color": "#fef9e7"},
                {"range": [65, 80], "color": "#eafaf1"},
                {"range": [80, 100], "color": "#d5f5e3"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": score
            }
        },
        title={"text": "Portfolio Health Score", "font": {"size": 20}},
    ))

    fig.update_layout(height=300, template="plotly_white", margin=dict(t=60, b=20))
    return fig


# ---------------------------------------------------------------------------
# 7. Score Breakdown Bar Chart
# ---------------------------------------------------------------------------

def plot_score_breakdown(components: dict) -> go.Figure:
    """
    Horizontal bar chart showing each component's score contribution.
    components: dict from portfolio_health_score()['components']
    """
    labels = list(components.keys())
    scores = [components[k]["score"] for k in labels]
    colors = ["#2ecc71" if s >= 65 else "#e67e22" if s >= 40 else "#e74c3c" for s in scores]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{s:.0f}/100" for s in scores],
        textposition="outside",
        hovertemplate="%{y}: %{x:.0f}/100<extra></extra>"
    ))

    fig.update_layout(
        title="Health Score Breakdown",
        xaxis=dict(range=[0, 115], title="Score"),
        yaxis=dict(title=""),
        template="plotly_white",
        height=300,
        margin=dict(l=160),
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Risk-Return Scatter Plot
# ---------------------------------------------------------------------------

def plot_risk_return_scatter(
    ann_returns: pd.Series,
    ann_volatility: pd.Series,
    sharpe_series: pd.Series,
    benchmark_return: float = None,
    benchmark_vol: float = None,
    fund_names: dict = None,
) -> go.Figure:
    """
    Scatter plot of Annualised Return (Y) vs Annualised Volatility (X) for each fund.
    Bubble size = Sharpe Ratio (larger = better risk-adjusted return).
    Nifty 50 plotted as a distinct reference point.

    Parameters
    ----------
    ann_returns    : pd.Series — annualised return per fund (scheme codes as index)
    ann_volatility : pd.Series — annualised volatility per fund
    sharpe_series  : pd.Series — Sharpe ratio per fund (used for bubble size)
    benchmark_return : float  — Nifty 50 annualised return (optional)
    benchmark_vol    : float  — Nifty 50 annualised volatility (optional)
    fund_names     : dict {scheme_code: name}
    """
    fig = go.Figure()
    color_map = _fund_color_map(ann_returns.index.tolist())

    # Normalise Sharpe to bubble size (min 12, max 40)
    sharpe_vals  = sharpe_series.reindex(ann_returns.index).fillna(0)
    sharpe_min   = sharpe_vals.min()
    sharpe_max   = sharpe_vals.max()
    sharpe_range = sharpe_max - sharpe_min if sharpe_max != sharpe_min else 1
    bubble_sizes = 12 + ((sharpe_vals - sharpe_min) / sharpe_range) * 28

    # --- Fund dots ---
    for code in ann_returns.index:
        label   = fund_names.get(code, code)[:45] if fund_names else code
        ret_pct = ann_returns[code] * 100
        vol_pct = ann_volatility[code] * 100
        sharpe  = sharpe_series.get(code, float("nan"))
        size    = bubble_sizes[code]

        fig.add_trace(go.Scatter(
            x=[vol_pct],
            y=[ret_pct],
            mode="markers+text",
            name=label,
            marker=dict(
                size=size,
                color=color_map[code],
                line=dict(width=1.5, color="white"),
                opacity=0.85,
            ),
            text=[label[:20]],
            textposition="top center",
            textfont=dict(size=10),
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Annualised Return: %{{y:.1f}}%<br>"
                f"Volatility: %{{x:.1f}}%<br>"
                f"Sharpe Ratio: {sharpe:.2f}<br>"
                "<extra></extra>"
            ),
        ))

    # --- Nifty 50 reference point ---
    if benchmark_return is not None and benchmark_vol is not None:
        fig.add_trace(go.Scatter(
            x=[benchmark_vol * 100],
            y=[benchmark_return * 100],
            mode="markers+text",
            name="Nifty 50",
            marker=dict(
                size=20,
                color="#2c3e50",
                symbol="diamond",
                line=dict(width=2, color="white"),
            ),
            text=["Nifty 50"],
            textposition="top center",
            textfont=dict(size=10, color="#2c3e50"),
            hovertemplate=(
                "<b>Nifty 50 (Benchmark)</b><br>"
                f"Annualised Return: {benchmark_return*100:.1f}%<br>"
                f"Volatility: {benchmark_vol*100:.1f}%<br>"
                "<extra></extra>"
            ),
        ))

    # --- Reference lines: risk-free rate (6%) horizontal ---
    fig.add_hline(
        y=6, line_dash="dot", line_color="#e74c3c", opacity=0.6,
        annotation_text="Risk-Free Rate (6%)",
        annotation_position="bottom right",
        annotation_font_size=10,
    )

    # --- Quadrant shading: top-left = ideal (high return, low risk) ---
    all_vols = list(ann_volatility * 100)
    if benchmark_vol:
        all_vols.append(benchmark_vol * 100)
    mid_vol = float(pd.Series(all_vols).mean())
    mid_ret = float(ann_returns.mean() * 100)

    fig.add_vrect(
        x0=0, x1=mid_vol,
        fillcolor="#2ecc71", opacity=0.04,
        annotation_text="Lower Risk →", annotation_position="top left",
        annotation_font_size=9, annotation_font_color="#27ae60",
    )

    fig.update_layout(
        title=dict(text="Risk-Return Map", font=dict(size=18)),
        xaxis=dict(
            title="Annualised Volatility (Risk) →",
            ticksuffix="%",
            showgrid=True, gridcolor="#f0f0f0",
            zeroline=False,
        ),
        yaxis=dict(
            title="← Annualised Return",
            ticksuffix="%",
            showgrid=True, gridcolor="#f0f0f0",
            zeroline=False,
        ),
        hovermode="closest",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        template="plotly_white",
        height=500,
        margin=dict(t=80),
    )

    # Bubble size legend note
    fig.add_annotation(
        text="Bubble size = Sharpe Ratio",
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        showarrow=False,
        font=dict(size=10, color="gray"),
    )

    return fig


# ---------------------------------------------------------------------------
# 9. SIP — Invested vs Corpus Chart
# ---------------------------------------------------------------------------

def plot_sip_corpus(corpus_df: pd.DataFrame, invested_df: pd.DataFrame, fund_names: dict = None) -> go.Figure:
    """
    Dual-line chart per fund: cumulative invested amount vs actual portfolio value.
    corpus_df / invested_df: DataFrames with DatetimeIndex, cols = scheme codes.
    """
    fig = go.Figure()
    color_map = _fund_color_map(corpus_df.columns.tolist())

    for col in corpus_df.columns:
        label = fund_names.get(col, col)[:40] if fund_names else col
        color = color_map[col]

        # Corpus (actual value) — solid line
        fig.add_trace(go.Scatter(
            x=corpus_df.index,
            y=corpus_df[col],
            mode="lines",
            name=f"{label} — Value",
            line=dict(color=color, width=2),
            hovertemplate="%{x|%b %Y}<br>Portfolio Value: ₹%{y:,.0f}<extra>" + label + "</extra>"
        ))

        # Invested — dashed line same color
        if col in invested_df.columns:
            fig.add_trace(go.Scatter(
                x=invested_df.index,
                y=invested_df[col],
                mode="lines",
                name=f"{label} — Invested",
                line=dict(color=color, width=1.5, dash="dash"),
                hovertemplate="%{x|%b %Y}<br>Invested: ₹%{y:,.0f}<extra>" + label + " (invested)</extra>"
            ))

    fig.update_layout(
        title="SIP: Portfolio Value vs Amount Invested",
        xaxis_title="Date",
        yaxis_title="Amount (₹)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=450,
        yaxis=dict(tickformat=",.0f", tickprefix="₹"),
    )
    return fig
