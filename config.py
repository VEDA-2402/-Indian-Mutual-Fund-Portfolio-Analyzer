import os
from dotenv import load_dotenv

load_dotenv()

# --- Risk-Free Rate (Indian T-bill proxy) ---
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", 0.06))  # 6% annually

# --- Benchmark ---
BENCHMARK_TICKER = os.getenv("BENCHMARK_TICKER", "^NSEI")  # Nifty 50

# --- Data Fetch Settings ---
NAV_CACHE_TTL_SECONDS = int(os.getenv("NAV_CACHE_TTL_SECONDS", 3600))  # 1 hour cache

# --- Rolling Return Windows (in trading days) ---
ROLLING_WINDOWS = {
    "1M":  21,
    "3M":  63,
    "6M":  126,
    "1Y":  252,
    "3Y":  756,
    "5Y":  1260,
}

# --- Portfolio Health Score Weights ---
SCORE_WEIGHTS = {
    "sharpe":      0.30,   # rewards good risk-adjusted return
    "drawdown":    0.25,   # penalizes large drawdowns
    "correlation": 0.20,   # penalizes high inter-fund correlation
    "returns":     0.25,   # rewards consistent returns
}

# --- App Settings ---
APP_TITLE = "Indian Mutual Fund Portfolio Analyzer"
APP_ICON  = "📈"
