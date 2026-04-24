"""
data/cache_manager.py
---------------------
Hybrid disk cache for NAV and fund list data.
- Saves CSV files locally under data/cache/
- Auto-expires after CACHE_TTL_HOURS (default 24h)
- Force-refresh clears the cache and re-fetches from source
- UI-agnostic: no Streamlit imports
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE_DIR       = Path(__file__).parent / "cache"
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", 24))
META_FILE       = CACHE_DIR / "_meta.json"  # tracks last-fetched timestamps

CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Metadata Helpers
# ---------------------------------------------------------------------------

def _load_meta() -> dict:
    if META_FILE.exists():
        try:
            with open(META_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_meta(meta: dict):
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)


def _is_expired(key: str) -> bool:
    """Returns True if cache for `key` is missing or older than TTL."""
    meta = _load_meta()
    if key not in meta:
        return True
    last_fetched = datetime.fromisoformat(meta[key])
    return datetime.now() - last_fetched > timedelta(hours=CACHE_TTL_HOURS)


def _mark_fetched(key: str):
    meta = _load_meta()
    meta[key] = datetime.now().isoformat()
    _save_meta(meta)


def _cache_path(key: str) -> Path:
    """Returns the CSV path for a given cache key."""
    safe_key = key.replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{safe_key}.csv"


# ---------------------------------------------------------------------------
# Fund List Cache
# ---------------------------------------------------------------------------

FUND_LIST_KEY = "fund_list"
FUND_LIST_PATH = CACHE_DIR / "fund_list.json"


def get_cached_fund_list() -> Optional[dict]:
    """Returns cached fund list dict if fresh, else None."""
    if _is_expired(FUND_LIST_KEY):
        return None
    if not FUND_LIST_PATH.exists():
        return None
    try:
        with open(FUND_LIST_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None


def save_fund_list(funds: dict):
    """Saves fund list dict to disk and updates metadata."""
    with open(FUND_LIST_PATH, "w") as f:
        json.dump(funds, f)
    _mark_fetched(FUND_LIST_KEY)


# ---------------------------------------------------------------------------
# NAV Cache
# ---------------------------------------------------------------------------

def get_cached_nav(scheme_code: str) -> Optional[pd.Series]:
    """Returns cached NAV Series if fresh, else None."""
    key  = f"nav_{scheme_code}"
    path = _cache_path(key)

    if _is_expired(key) or not path.exists():
        return None

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        series = df.iloc[:, 0].rename(scheme_code)
        series.index = pd.to_datetime(series.index)
        return series
    except Exception:
        return None


def save_nav(scheme_code: str, series: pd.Series):
    """Saves NAV Series to CSV and updates metadata."""
    key  = f"nav_{scheme_code}"
    path = _cache_path(key)
    series.to_frame().to_csv(path)
    _mark_fetched(key)


# ---------------------------------------------------------------------------
# Benchmark Cache
# ---------------------------------------------------------------------------

def get_cached_benchmark() -> Optional[pd.Series]:
    """Returns cached Nifty 50 Series if fresh, else None."""
    key  = "benchmark_nifty50"
    path = _cache_path(key)

    if _is_expired(key) or not path.exists():
        return None

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        series = df.iloc[:, 0].rename("Nifty50")
        series.index = pd.to_datetime(series.index)
        return series
    except Exception:
        return None


def save_benchmark(series: pd.Series):
    """Saves Nifty 50 Series to CSV and updates metadata."""
    key  = "benchmark_nifty50"
    path = _cache_path(key)
    series.to_frame().to_csv(path)
    _mark_fetched(key)


# ---------------------------------------------------------------------------
# Force Refresh
# ---------------------------------------------------------------------------

def clear_all_cache():
    """Deletes all cached CSV/JSON files and resets metadata (including TER)."""
    for f in CACHE_DIR.glob("*.csv"):
        f.unlink(missing_ok=True)
    for f in CACHE_DIR.glob("*.json"):
        f.unlink(missing_ok=True)


def clear_nav_cache(scheme_code: str):
    """Clears cache only for a specific fund."""
    key  = f"nav_{scheme_code}"
    path = _cache_path(key)
    path.unlink(missing_ok=True)
    meta = _load_meta()
    meta.pop(key, None)
    _save_meta(meta)


# ---------------------------------------------------------------------------
# Cache Status (for UI display)
# ---------------------------------------------------------------------------

def get_cache_status() -> dict:
    """
    Returns a summary of what's currently cached and when it was last fetched.
    Used by the sidebar to show cache freshness info.
    """
    meta = _load_meta()
    status = {}
    for key, ts in meta.items():
        last_fetched = datetime.fromisoformat(ts)
        age_hours    = (datetime.now() - last_fetched).total_seconds() / 3600
        expires_in   = max(0, CACHE_TTL_HOURS - age_hours)
        status[key]  = {
            "last_fetched": last_fetched.strftime("%d %b %Y, %H:%M"),
            "age_hours":    round(age_hours, 1),
            "expires_in_h": round(expires_in, 1),
            "is_fresh":     expires_in > 0,
        }
    return status
