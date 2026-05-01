"""
report_generator.py
-------------------
Generates a clean, branded PDF portfolio analysis report using reportlab.
Called from the Streamlit UI — returns PDF bytes for st.download_button().

Sections:
  1. Cover — title, generated date, fund list
  2. Portfolio Summary — key metrics (Sharpe, return, volatility, drawdown)
  3. Individual Fund Metrics — table per fund
  4. Rolling Returns — 1M / 3M / 6M / 1Y / 3Y
  5. Benchmark Comparison — excess return vs Nifty 50
  6. Portfolio Health Score — total score + component breakdown
  7. Disclaimer
"""

import io
import numpy as np
import pandas as pd
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)

# ---------------------------------------------------------------------------
# Colour Palette
# ---------------------------------------------------------------------------

C_NAVY    = colors.HexColor("#0D1B2A")   # headers, title
C_BLUE    = colors.HexColor("#1A73E8")   # accent, section rules
C_LIGHT   = colors.HexColor("#F0F4FF")   # table row alternates
C_GREEN   = colors.HexColor("#1E8449")   # positive values
C_RED     = colors.HexColor("#C0392B")   # negative values
C_GRAY    = colors.HexColor("#6C757D")   # captions, footnotes
C_WHITE   = colors.white
C_BLACK   = colors.black


# ---------------------------------------------------------------------------
# Style Helpers
# ---------------------------------------------------------------------------

def _styles():
    base = getSampleStyleSheet()

    def S(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        "title": S("RPTitle",
            fontSize=22, textColor=C_NAVY, leading=28,
            spaceAfter=4, alignment=TA_CENTER, fontName="Helvetica-Bold"),
        "subtitle": S("RPSubtitle",
            fontSize=11, textColor=C_GRAY, leading=14,
            spaceAfter=2, alignment=TA_CENTER, fontName="Helvetica"),
        "section": S("RPSection",
            fontSize=13, textColor=C_NAVY, leading=18,
            spaceBefore=14, spaceAfter=4, fontName="Helvetica-Bold"),
        "normal": S("RPNormal",
            fontSize=9, textColor=C_BLACK, leading=13,
            spaceAfter=3, fontName="Helvetica"),
        "small": S("RPSmall",
            fontSize=8, textColor=C_GRAY, leading=11,
            spaceAfter=2, fontName="Helvetica"),
        "caption": S("RPCaption",
            fontSize=8, textColor=C_GRAY, leading=11,
            spaceAfter=6, alignment=TA_CENTER, fontName="Helvetica-Oblique"),
        "th": S("RPTH",
            fontSize=8, textColor=C_WHITE, leading=11,
            alignment=TA_CENTER, fontName="Helvetica-Bold"),
        "td": S("RPTD",
            fontSize=8, textColor=C_BLACK, leading=11,
            alignment=TA_CENTER, fontName="Helvetica"),
        "td_left": S("RPTDLeft",
            fontSize=8, textColor=C_BLACK, leading=11,
            alignment=TA_LEFT, fontName="Helvetica"),
        "score_big": S("RPScoreBig",
            fontSize=28, textColor=C_NAVY, leading=34,
            alignment=TA_CENTER, fontName="Helvetica-Bold"),
        "disclaimer": S("RPDisclaimer",
            fontSize=7.5, textColor=C_GRAY, leading=11,
            spaceAfter=2, fontName="Helvetica-Oblique"),
    }


def _rule():
    return HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=8)


def _section(title, st):
    return [Paragraph(title, st["section"]), _rule()]


def _fmt_pct(val, decimals=1):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


def _fmt_f(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def _fmt_inr(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"Rs {val:,.0f}"


def _color_cell(val, good_positive=True):
    """Return green for positive/good, red for negative/bad."""
    try:
        v = float(str(val).replace("%", "").replace("Rs", "").replace(",", "").strip())
        if good_positive:
            return C_GREEN if v >= 0 else C_RED
        else:
            return C_RED if v >= 0 else C_GREEN
    except Exception:
        return C_BLACK


def _table_style(header_rows=1, alt_color=True):
    cmds = [
        ("BACKGROUND",  (0, 0), (-1, header_rows - 1), C_NAVY),
        ("TEXTCOLOR",   (0, 0), (-1, header_rows - 1), C_WHITE),
        ("FONTNAME",    (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",       (0, 1), (0, -1), "LEFT"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUND",(0, header_rows), (-1, -1),
         [C_WHITE, C_LIGHT] if alt_color else [C_WHITE]),
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",(0, 0), (-1, -1), 6),
    ]
    return TableStyle(cmds)


# ---------------------------------------------------------------------------
# Section Builders
# ---------------------------------------------------------------------------

def _build_cover(fund_names: dict, date_range: str, st: dict) -> list:
    story = []
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph("Indian Mutual Fund", st["title"]))
    story.append(Paragraph("Portfolio Analysis Report", st["title"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"Generated on {datetime.today().strftime('%d %B %Y')}", st["subtitle"]))
    story.append(Paragraph(f"Analysis Period: {date_range}", st["subtitle"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(_rule())
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Funds in this Report", st["section"]))
    for code, name in fund_names.items():
        story.append(Paragraph(f"• {name}  <font color='#6C757D'>[{code}]</font>", st["normal"]))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "Data sourced from AMFI (via mftool) and NSE (via yfinance). "
        "Benchmark: Nifty 50.",
        st["small"]
    ))
    return story


def _build_summary(sharpe_s, ann_ret_s, ann_vol_s, mdd_s, fund_names: dict, st: dict) -> list:
    story = []
    story += _section("Portfolio Summary", st)

    headers = [
        Paragraph("Fund", st["th"]),
        Paragraph("Ann. Return", st["th"]),
        Paragraph("Volatility", st["th"]),
        Paragraph("Sharpe Ratio", st["th"]),
        Paragraph("Max Drawdown", st["th"]),
    ]
    rows = [headers]

    for code, name in fund_names.items():
        short_name = name[:40] + "…" if len(name) > 40 else name
        rows.append([
            Paragraph(short_name, st["td_left"]),
            Paragraph(_fmt_pct(ann_ret_s.get(code, np.nan)), st["td"]),
            Paragraph(_fmt_pct(ann_vol_s.get(code, np.nan)), st["td"]),
            Paragraph(_fmt_f(sharpe_s.get(code, np.nan)), st["td"]),
            Paragraph(_fmt_pct(mdd_s.get(code, np.nan)), st["td"]),
        ])

    col_widths = [7.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 2.8 * cm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(_table_style())
    story.append(t)
    return story


def _build_rolling_returns(latest_rolling_df: pd.DataFrame, fund_names: dict, st: dict) -> list:
    story = []
    story += _section("Rolling Returns", st)
    story.append(Paragraph(
        "Latest point-in-time rolling returns for each fund across standard windows.",
        st["small"]
    ))
    story.append(Spacer(1, 0.15 * cm))

    windows = [w for w in ["1M", "3M", "6M", "1Y", "3Y", "5Y"] if w in latest_rolling_df.index]
    if not windows:
        story.append(Paragraph("Insufficient data for rolling returns.", st["small"]))
        return story

    header = [Paragraph("Fund", st["th"])] + [Paragraph(w, st["th"]) for w in windows]
    rows = [header]

    for code, name in fund_names.items():
        if code not in latest_rolling_df.columns:
            continue
        short_name = name[:38] + "…" if len(name) > 38 else name
        row = [Paragraph(short_name, st["td_left"])]
        for w in windows:
            val = latest_rolling_df.loc[w, code] if w in latest_rolling_df.index else np.nan
            row.append(Paragraph(_fmt_pct(val), st["td"]))
        rows.append(row)

    n_windows = len(windows)
    col_widths = [6.5 * cm] + [((A4[0] - 4 * cm - 6.5 * cm) / n_windows)] * n_windows
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(_table_style())
    story.append(t)
    return story


def _build_benchmark(excess_ret, ann_ret_s, fund_names: dict, st: dict) -> list:
    """
    excess_ret is a pd.Series: {scheme_code: annualized_excess_return_vs_nifty50}.
    ann_ret_s  is a pd.Series: {scheme_code: annualized_return}.
    Both come directly from calculations/returns.py.
    """
    story = []
    story += _section("Benchmark Comparison vs Nifty 50", st)
    story.append(Paragraph(
        "Annualized excess return = Fund annualized return minus Nifty 50 annualized return "
        "over the selected period. Positive values indicate outperformance.",
        st["small"]
    ))
    story.append(Spacer(1, 0.15 * cm))

    # excess_ret may be a Series or DataFrame — normalise to Series
    if isinstance(excess_ret, pd.DataFrame):
        # Shouldn't happen with current code, but handle gracefully
        if excess_ret.empty:
            story.append(Paragraph("Benchmark data unavailable.", st["small"]))
            return story
        excess_series = excess_ret.iloc[-1]  # take last row
    elif isinstance(excess_ret, pd.Series):
        excess_series = excess_ret
    else:
        story.append(Paragraph("Benchmark data unavailable.", st["small"]))
        return story

    if excess_series.empty or excess_series.isna().all():
        story.append(Paragraph("Benchmark data unavailable.", st["small"]))
        return story

    header = [
        Paragraph("Fund", st["th"]),
        Paragraph("Fund Ann. Return", st["th"]),
        Paragraph("Excess vs Nifty 50", st["th"]),
        Paragraph("Verdict", st["th"]),
    ]
    rows = [header]

    for code, name in fund_names.items():
        short_name = name[:40] + "..." if len(name) > 40 else name
        fund_ret   = ann_ret_s.get(code, np.nan) if hasattr(ann_ret_s, 'get') else np.nan
        excess_val = excess_series.get(code, np.nan) if hasattr(excess_series, 'get') else np.nan

        verdict = "N/A"
        if not np.isnan(excess_val):
            verdict = "Outperformed" if excess_val >= 0 else "Underperformed"

        rows.append([
            Paragraph(short_name, st["td_left"]),
            Paragraph(_fmt_pct(fund_ret), st["td"]),
            Paragraph(_fmt_pct(excess_val), st["td"]),
            Paragraph(verdict, st["td"]),
        ])

    col_widths = [8 * cm, 3 * cm, 3.5 * cm, 3.3 * cm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(_table_style())
    story.append(t)
    return story


def _build_health_score(health: dict, individual_health: list, fund_names: dict, st: dict) -> list:
    story = []
    story += _section("Portfolio Health Score", st)

    # Overall score box
    grade = health.get("grade", "N/A")
    score = health.get("total_score", 0)
    interp = health.get("interpretation", "")

    score_data = [[
        Paragraph(f"{score:.0f} / 100", st["score_big"]),
        Paragraph(f"Grade: {grade}", st["section"]),
    ]]
    score_table = Table(score_data, colWidths=[6 * cm, 11.8 * cm])
    score_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (0, 0), C_LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, C_BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(interp, st["normal"]))
    story.append(Spacer(1, 0.3 * cm))

    # Component breakdown
    story.append(Paragraph("Score Breakdown", st["section"]))
    components = health.get("components", {})
    comp_rows = [[
        Paragraph("Component", st["th"]),
        Paragraph("Score", st["th"]),
        Paragraph("Raw Value", st["th"]),
        Paragraph("Description", st["th"]),
    ]]
    for comp_name, comp_data in components.items():
        comp_rows.append([
            Paragraph(comp_name, st["td_left"]),
            Paragraph(f"{comp_data['score']:.0f} / 100", st["td"]),
            Paragraph(str(comp_data.get("raw_value", "N/A")), st["td"]),
            Paragraph(comp_data.get("label", ""), st["td_left"]),
        ])
    comp_table = Table(comp_rows, colWidths=[4.5*cm, 2.5*cm, 3*cm, 7.8*cm], repeatRows=1)
    comp_table.setStyle(_table_style())
    story.append(comp_table)
    story.append(Spacer(1, 0.3 * cm))

    # Individual fund scores
    if individual_health:
        story.append(Paragraph("Individual Fund Scores", st["section"]))
        ind_rows = [[
            Paragraph("Fund", st["th"]),
            Paragraph("Score", st["th"]),
            Paragraph("Grade", st["th"]),
        ]]
        for fs in individual_health:
            code = fs.get("scheme_code", "")
            name = fund_names.get(code, code)
            short_name = name[:45] + "…" if len(name) > 45 else name
            ind_rows.append([
                Paragraph(short_name, st["td_left"]),
                Paragraph(f"{fs['total_score']:.0f} / 100", st["td"]),
                Paragraph(fs.get("grade", "N/A"), st["td"]),
            ])
        ind_table = Table(ind_rows, colWidths=[12*cm, 3.5*cm, 2.3*cm], repeatRows=1)
        ind_table.setStyle(_table_style())
        story.append(ind_table)

    return story


def _build_nav_snapshot(nav_snapshot_df, fund_names: dict, st: dict) -> list:
    story = []
    story += _section('NAV Snapshot', st)
    story.append(Paragraph(
        'Latest NAV, 52-week high and low, and how far the current NAV sits from its peak and trough.',
        st['small']
    ))
    story.append(Spacer(1, 0.15 * cm))
    if nav_snapshot_df is None or nav_snapshot_df.empty:
        story.append(Paragraph('NAV snapshot data unavailable.', st['small']))
        return story
    header = [
        Paragraph('Fund', st['th']),
        Paragraph('Latest NAV', st['th']),
        Paragraph('52W High', st['th']),
        Paragraph('52W Low', st['th']),
        Paragraph('From High', st['th']),
        Paragraph('From Low', st['th']),
    ]
    rows = [header]
    for code, name in fund_names.items():
        if code not in nav_snapshot_df.index:
            continue
        d = nav_snapshot_df.loc[code]
        short_name = name[:35] + '...' if len(name) > 35 else name
        def safe_pct(v):
            try:
                return _fmt_pct(float(v) / 100)
            except Exception:
                return 'N/A'
        rows.append([
            Paragraph(short_name, st['td_left']),
            Paragraph(str(d.get('latest_nav', 'N/A')), st['td']),
            Paragraph(str(d.get('week52_high', 'N/A')), st['td']),
            Paragraph(str(d.get('week52_low',  'N/A')), st['td']),
            Paragraph(safe_pct(d.get('pct_from_high', 'nan')), st['td']),
            Paragraph(safe_pct(d.get('pct_from_low',  'nan')), st['td']),
        ])
    col_widths = [5.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.3*cm]
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    t.setStyle(_table_style())
    story.append(t)
    return story


def _build_correlation(corr_matrix, fund_names: dict, st: dict) -> list:
    story = []
    story += _section('Fund Correlation Matrix', st)
    story.append(Paragraph(
        'Pairwise correlation of daily returns. Values close to 1.0 mean funds move together '
        '(low diversification). Values below 0.6 indicate meaningful diversification benefit.',
        st['small']
    ))
    story.append(Spacer(1, 0.15 * cm))
    if corr_matrix is None or corr_matrix.empty:
        story.append(Paragraph('Correlation data unavailable.', st['small']))
        return story
    codes = [c for c in fund_names.keys() if c in corr_matrix.index]
    if not codes:
        story.append(Paragraph('Correlation data unavailable.', st['small']))
        return story
    short_labels = {}
    for code in codes:
        name = fund_names.get(code, code)
        words = name.split()
        short_labels[code] = ' '.join(words[:2]) if len(words) >= 2 else name[:12]
    header = [Paragraph('Fund', st['th'])] + [Paragraph(short_labels[c], st['th']) for c in codes]
    rows = [header]
    for row_code in codes:
        row = [Paragraph(short_labels[row_code], st['td_left'])]
        for col_code in codes:
            try:
                cell_text = f"{float(corr_matrix.loc[row_code, col_code]):.2f}"
            except Exception:
                cell_text = 'N/A'
            row.append(Paragraph(cell_text, st['td']))
        rows.append(row)
    n_cols = len(codes)
    label_width = 4.5 * cm
    cell_w = (A4[0] - 4 * cm - label_width) / max(n_cols, 1)
    col_widths = [label_width] + [cell_w] * n_cols
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    ts = _table_style()
    for i in range(1, len(rows)):
        ts.add('BACKGROUND', (i, i), (i, i), colors.HexColor('#D6EAF8'))
    t.setStyle(ts)
    story.append(t)
    return story


def _build_diagnosis(fund_names, sharpe_s, ann_ret_s, mdd_s, excess_ret,
                     avg_corr, health, benchmark_ann_ret, st) -> list:
    story = []
    story += _section('Portfolio Diagnosis', st)
    lines = []
    try:
        if not ann_ret_s.isna().all():
            best_code  = ann_ret_s.idxmax()
            worst_code = ann_ret_s.idxmin()
            best_name  = fund_names.get(best_code, best_code)
            worst_name = fund_names.get(worst_code, worst_code)
            if best_code != worst_code:
                lines.append(
                    f"{best_name} delivered the strongest annualized return of "
                    f"{ann_ret_s[best_code]*100:.1f}%, while {worst_name} returned "
                    f"{ann_ret_s[worst_code]*100:.1f}% annually."
                )
            else:
                lines.append(
                    f"The portfolio holds {best_name} with an annualized return of "
                    f"{ann_ret_s[best_code]*100:.1f}%."
                )
    except Exception:
        pass
    try:
        bench_ret = float(benchmark_ann_ret)
        avg_ret = ann_ret_s.mean()
        direction = 'outperformed' if avg_ret > bench_ret else 'underperformed'
        note = 'a positive sign.' if avg_ret > bench_ret else 'Consider reviewing the fund selection.'
        lines.append(
            f"On average, the portfolio {direction} the Nifty 50 benchmark "
            f"({avg_ret*100:.1f}% vs {bench_ret*100:.1f}% annualized). {note}"
        )
    except Exception:
        pass
    try:
        if not mdd_s.isna().all():
            avg_dd = mdd_s.mean()
            worst_code = mdd_s.idxmin()
            worst_name = fund_names.get(worst_code, worst_code)
            dd_label = 'moderate' if abs(avg_dd) < 0.25 else 'high'
            lines.append(
                f"The portfolio has {dd_label} downside risk with an average max drawdown of "
                f"{avg_dd*100:.1f}%. {worst_name} experienced the steepest decline at "
                f"{mdd_s[worst_code]*100:.1f}%."
            )
    except Exception:
        pass
    try:
        corr_val = float(avg_corr)
        if not np.isnan(corr_val):
            if corr_val < 0.6:
                lines.append(
                    f"With an average inter-fund correlation of {corr_val:.2f}, the portfolio is "
                    f"reasonably diversified -- the funds do not move in lockstep."
                )
            else:
                lines.append(
                    f"The average inter-fund correlation is {corr_val:.2f}, which is relatively high. "
                    f"The funds tend to move together, limiting the diversification benefit."
                )
    except Exception:
        pass
    try:
        score = health.get('total_score', 0)
        grade = health.get('grade', 'N/A')
        interp = health.get('interpretation', '')
        lines.append(
            f"Overall, the portfolio scores {score:.0f}/100 (Grade {grade}) on the "
            f"Portfolio Health Score. {interp}"
        )
    except Exception:
        pass
    for line in lines:
        story.append(Paragraph(line, st['normal']))
        story.append(Spacer(1, 0.15 * cm))
    return story


def _build_disclaimer(st: dict) -> list:
    story = []
    story.append(Spacer(1, 0.5 * cm))
    story += _section('Disclaimer', st)
    story.append(Paragraph(
        'This report is generated for informational and educational purposes only. '
        'All data is sourced from AMFI (Association of Mutual Funds in India) via mftool '
        'and NSE (National Stock Exchange) via yfinance. '
        'Past performance of mutual funds is not indicative of future results. '
        'The Portfolio Health Score and fund recommendations are based on historical metrics '
        'and do not constitute financial advice. '
        'Please consult a SEBI-registered investment advisor before making any investment decisions. '
        'Mutual fund investments are subject to market risks. Please read all scheme-related '
        'documents carefully before investing.',
        st['disclaimer']
    ))
    return story


def generate_pdf_report(
    fund_names: dict,
    sharpe_s: pd.Series,
    ann_ret_s: pd.Series,
    ann_vol_s: pd.Series,
    mdd_s: pd.Series,
    latest_rolling_df: pd.DataFrame,
    excess_ret,
    health: dict,
    individual_health: list,
    nav_snapshot_df=None,
    corr_matrix=None,
    avg_corr: float = float('nan'),
    benchmark_ann_ret: float = 0.0,
    date_range: str = 'Full Available History',
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title='MF Portfolio Analysis Report',
        author='Indian MF Portfolio Analyzer',
    )
    st = _styles()
    story = []
    story += _build_cover(fund_names, date_range, st)
    story.append(PageBreak())
    story += _build_diagnosis(
        fund_names=fund_names, sharpe_s=sharpe_s, ann_ret_s=ann_ret_s,
        mdd_s=mdd_s, excess_ret=excess_ret, avg_corr=avg_corr,
        health=health, benchmark_ann_ret=benchmark_ann_ret, st=st,
    )
    story.append(Spacer(1, 0.4*cm))
    story += _build_summary(sharpe_s, ann_ret_s, ann_vol_s, mdd_s, fund_names, st)
    story.append(Spacer(1, 0.4*cm))
    story += _build_nav_snapshot(nav_snapshot_df, fund_names, st)
    story.append(PageBreak())
    story += _build_rolling_returns(latest_rolling_df, fund_names, st)
    story.append(Spacer(1, 0.4*cm))
    story += _build_benchmark(excess_ret, ann_ret_s, fund_names, st)
    story.append(Spacer(1, 0.4*cm))
    story += _build_correlation(corr_matrix, fund_names, st)
    story.append(PageBreak())
    story += _build_health_score(health, individual_health, fund_names, st)
    story += _build_disclaimer(st)
    doc.build(story)
    buffer.seek(0)
    return buffer.read()