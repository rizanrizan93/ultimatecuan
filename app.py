
import concurrent.futures as cf
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema, hilbert, periodogram

# =========================================================
# IDX / IHSG DUAL TAB SCANNER - FINAL VERSION
# Tab 1: Market Structure Top 20 + reversal signals
# Tab 2: Institutional Forward Score with sub-tabs + entry plan / benchmark / time analysis
# =========================================================

st.set_page_config(page_title="IDX Dual Tab Scanner", layout="wide")
st.title("📊 IDX Dual Tab Scanner")
st.caption(
    "Global watchlist untuk ranking cepat, lalu deep dive untuk bedah detail per ticker dengan institutional forward score, entry plan, dan time analysis."
)
st.markdown("---")

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("🎯 Universe Source")
universe_mode = st.sidebar.radio(
    "Pilih sumber universe",
    ["Paste tickers", "Upload CSV", "Local file midcap_universe.csv"],
    index=0,
)

paste_text = ""
uploaded_file = None
if universe_mode == "Paste tickers":
    paste_text = st.sidebar.text_area(
        "Paste tickers (satu per baris / dipisah koma)",
        value="BMRI\nBBCA\nTLKM\nASII",
        height=140,
    )
elif universe_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV universe", type=["csv"])
else:
    st.sidebar.info("Mode ini akan membaca file `midcap_universe.csv` dari folder aplikasi.")

st.sidebar.markdown("---")
st.sidebar.header("🧭 Scan Settings")
months = st.sidebar.slider("Periode data historis (bulan)", 12, 60, 24)
min_price = st.sidebar.number_input("Min harga (Rp)", value=200.0, step=10.0)
max_price = st.sidebar.number_input("Max harga (Rp)", value=25000.0, step=500.0)
min_avg_volume = st.sidebar.number_input("Min rata-rata volume 20D", value=250000, step=50000)
min_history_bars = st.sidebar.slider("Min candle valid", 60, 240, 100)

st.sidebar.markdown("---")
st.sidebar.header("🚀 Execution")
max_workers = st.sidebar.slider("Max parallel workers", 2, 12, 6)
run_global_scan = st.sidebar.button("Run global scan", type="primary")

GLOBAL_MODE = "Balanced"

# =========================================================
# Utilities
# =========================================================
def normalize_ticker(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if not s or s == "NAN":
        return ""
    if s.startswith("^"):
        return s
    return s if s.endswith(".JK") else f"{s}.JK"


def make_flow_score(flow_mode: str) -> float:
    mapping = {
        "Big Akumulasi": 95.0,
        "Small Akumulasi": 75.0,
        "Netral": 50.0,
        "Small Distribusi": 30.0,
        "Big Distribusi": 10.0,
    }
    return mapping.get(flow_mode, 50.0)


@st.cache_data(ttl=1800, show_spinner=False)
def load_ticker_data(symbol: str, months: int) -> pd.DataFrame:
    end = pd.Timestamp.now(tz=None)
    start = end - pd.DateOffset(months=months)

    base = str(symbol).strip()
    candidates = []
    if base:
        candidates.append(base)
        if base.endswith(".JK"):
            candidates.append(base[:-3])
        elif not base.startswith("^"):
            candidates.append(f"{base}.JK")

    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for candidate in candidates:
        for attempt in range(3):
            try:
                df = yf.download(
                    candidate,
                    period=f"{months}mo",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                if df is None or df.empty:
                    df = yf.download(
                        candidate,
                        start=start,
                        end=end,
                        auto_adjust=False,
                        progress=False,
                        threads=False,
                    )
            except Exception:
                df = pd.DataFrame()

            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    if any(col in df.columns.get_level_values(0) for col in ["Close", "Open", "High", "Low"]):
                        df.columns = df.columns.get_level_values(0)
                    else:
                        df.columns = df.columns.get_level_values(1)

                needed = {"Open", "High", "Low", "Close", "Volume"}
                if needed.issubset(set(df.columns)):
                    out = df.dropna().copy()
                    if not out.empty:
                        return out

            time.sleep(0.25 * (attempt + 1))

    return pd.DataFrame()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def zlema(series: pd.Series, span: int) -> pd.Series:
    """Zero-lag EMA approximation using a de-lagged input series."""
    s = series.astype(float).copy()
    if s.empty:
        return s
    lag = max(1, (int(span) - 1) // 2)
    lagged = s.shift(lag)
    de_lagged = s + (s - lagged)
    return de_lagged.ewm(span=span, adjust=False, min_periods=1).mean()

def highpass_filter(series: pd.Series, period: int = 48) -> pd.Series:
    """Causal 2-pole high-pass filter for low-lag detrending."""
    s = series.astype(float).ffill().bfill()
    if s.empty:
        return s
    period = int(max(10, period))
    # Ehlers-style coefficient; stable for trend extraction on daily data.
    denom = np.cos(0.707 * 2 * np.pi / period)
    if abs(denom) < 1e-9:
        denom = 1e-9
    alpha = (
        np.cos(0.707 * 2 * np.pi / period)
        + np.sin(0.707 * 2 * np.pi / period)
        - 1
    ) / denom

    vals = s.to_numpy(dtype=float)
    hp = np.zeros(len(vals), dtype=float)

    if len(vals) < 3:
        return pd.Series(hp, index=s.index)

    a1 = (1 - alpha / 2.0) ** 2
    b1 = 2 * (1 - alpha)
    b2 = (1 - alpha) ** 2

    for i in range(2, len(vals)):
        hp[i] = (
            a1 * (vals[i] - 2 * vals[i - 1] + vals[i - 2])
            + b1 * hp[i - 1]
            - b2 * hp[i - 2]
        )

    return pd.Series(hp, index=s.index)


def linear_forecast_pad(arr: np.ndarray, n_future: int = 12, fit_points: int = 20) -> np.ndarray:
    """Append a small linear forecast to reduce Hilbert edge distortion on the last bar."""
    x = np.asarray(arr, dtype=float)
    if x.size < 3 or n_future <= 0:
        return x.copy()

    fit_points = int(max(3, min(fit_points, x.size)))
    y = x[-fit_points:]
    idx = np.arange(fit_points, dtype=float)

    try:
        slope, intercept = np.polyfit(idx, y, 1)
        future_idx = np.arange(fit_points, fit_points + int(n_future), dtype=float)
        future = slope * future_idx + intercept
        return np.concatenate([x, future])
    except Exception:
        return x.copy()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series):
    macd_line = ema(close, 12) - ema(close, 26)
    signal_line = ema(macd_line, 9)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr_w = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = (
        100
        * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        / atr_w
    )
    minus_di = (
        100
        * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        / atr_w
    )

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid, mid + std_mult * std, mid - std_mult * std


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0.0)
    return (direction * df["Volume"].fillna(0.0)).cumsum()


def chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"].fillna(0.0)

    price_range = (high - low).replace(0, np.nan)
    mfm = (((close - low) - (high - close)) / price_range).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mfv = mfm * volume
    cmf = mfv.rolling(period, min_periods=period).sum() / volume.rolling(period, min_periods=period).sum().replace(0, np.nan)
    return cmf


def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    raw_money_flow = typical_price * df["Volume"].fillna(0.0)
    delta = typical_price.diff()
    positive_mf = raw_money_flow.where(delta > 0, 0.0)
    negative_mf = raw_money_flow.where(delta < 0, 0.0).abs()
    pos_sum = positive_mf.rolling(period, min_periods=period).sum()
    neg_sum = negative_mf.rolling(period, min_periods=period).sum().replace(0, np.nan)
    mfr = pos_sum / neg_sum
    return 100 - (100 / (1 + mfr))


def stochastic_oscillator(df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> tuple[pd.Series, pd.Series]:
    low_min = df["Low"].rolling(period, min_periods=period).min()
    high_max = df["High"].rolling(period, min_periods=period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100 * (df["Close"] - low_min) / denom
    k = k.rolling(smooth_k, min_periods=1).mean()
    d = k.rolling(smooth_d, min_periods=1).mean()
    return k, d


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    sma = tp.rolling(period, min_periods=period).mean()
    mad = (tp - sma).abs().rolling(period, min_periods=period).mean()
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def rate_of_change(close: pd.Series, period: int = 12) -> pd.Series:
    return close.pct_change(periods=period) * 100


@st.cache_data(ttl=86400, show_spinner=False)
def load_yf_info(symbol: str) -> dict:
    """Cached Yahoo Finance info fetch.

    yfinance .info is slow and sometimes flaky; this wrapper centralizes the request
    and keeps the rest of the code from issuing duplicate calls.
    """
    base = str(symbol).strip()
    if not base:
        return {}
    try:
        info = yf.Ticker(base).info or {}
    except Exception:
        info = {}
    return info


@st.cache_data(ttl=86400, show_spinner=False)
def load_fundamental_snapshot(symbol: str) -> dict:
    out = {
        "peg_ratio": np.nan,
        "trailing_pe": np.nan,
        "forward_pe": np.nan,
        "revenue_growth": np.nan,
        "earnings_growth": np.nan,
        "profit_margins": np.nan,
    }
    base = str(symbol).strip()
    if not base:
        return out

    info = load_yf_info(base)

    def pick(*keys):
        for key in keys:
            val = info.get(key)
            if val is not None and val != "":
                try:
                    return float(val)
                except Exception:
                    continue
        return np.nan

    out["peg_ratio"] = pick("pegRatio", "peg_ratio")
    out["trailing_pe"] = pick("trailingPE", "trailing_pe")
    out["forward_pe"] = pick("forwardPE", "forward_pe")
    out["revenue_growth"] = pick("revenueGrowth")
    out["earnings_growth"] = pick("earningsGrowth", "earningsQuarterlyGrowth")
    out["profit_margins"] = pick("profitMargins")
    return out



@st.cache_data(ttl=86400, show_spinner=False)
def compute_fundamental_grade(symbol: str) -> dict:
    snap = load_fundamental_snapshot(symbol).copy()
    base = str(symbol).strip()
    if not base:
        snap.update(
            {
                "market_cap": np.nan,
                "current_ratio": np.nan,
                "debt_to_equity": np.nan,
                "return_on_equity": np.nan,
                "return_on_assets": np.nan,
                "operating_margin": np.nan,
                "gross_margin": np.nan,
                "free_cashflow": np.nan,
                "operating_cashflow": np.nan,
                "fundamental_score": np.nan,
                "growth_score": np.nan,
                "quality_score": np.nan,
                "health_score": np.nan,
                "valuation_score": np.nan,
                "fundamental_grade": "n/a",
            }
        )
        return snap

    info = load_yf_info(base)

    def pick(*keys):
        for key in keys:
            val = info.get(key)
            if val is not None and val != "":
                try:
                    return float(val)
                except Exception:
                    continue
        return np.nan

    def pct_like(v):
        if v is None or pd.isna(v):
            return np.nan
        v = float(v)
        return v if abs(v) <= 1.5 else v / 100.0

    def norm(v, lo, hi, invert=False):
        if v is None or pd.isna(v):
            return np.nan
        if hi == lo:
            return 0.5
        x = (float(v) - lo) / (hi - lo)
        x = float(np.clip(x, 0.0, 1.0))
        return 1.0 - x if invert else x

    snap["market_cap"] = pick("marketCap")
    snap["current_ratio"] = pick("currentRatio")
    snap["debt_to_equity"] = pick("debtToEquity")
    snap["return_on_equity"] = pick("returnOnEquity")
    snap["return_on_assets"] = pick("returnOnAssets")
    snap["operating_margin"] = pick("operatingMargins")
    snap["gross_margin"] = pick("grossMargins")
    snap["free_cashflow"] = pick("freeCashflow")
    snap["operating_cashflow"] = pick("operatingCashflow")

    rev_g = pct_like(snap.get("revenue_growth"))
    earn_g = pct_like(snap.get("earnings_growth"))
    profit_m = pct_like(snap.get("profit_margins"))
    roe = pct_like(snap.get("return_on_equity"))
    roa = pct_like(snap.get("return_on_assets"))
    op_m = pct_like(snap.get("operating_margin"))
    gross_m = pct_like(snap.get("gross_margin"))
    cr = snap.get("current_ratio")
    dte = snap.get("debt_to_equity")
    fcf = snap.get("free_cashflow")
    ocf = snap.get("operating_cashflow")

    peg = snap.get("peg_ratio")
    trailing_pe = snap.get("trailing_pe")
    forward_pe = snap.get("forward_pe")
    if (pd.isna(peg) or not np.isfinite(float(peg))) and np.isfinite(forward_pe) and np.isfinite(earn_g):
        if float(earn_g) > 0:
            peg = float(forward_pe) / (float(earn_g) * 100.0 if abs(float(earn_g)) <= 1.5 else float(earn_g))
    snap["peg_ratio"] = peg

    growth_score = 50.0
    if np.isfinite(rev_g):
        growth_score += norm(rev_g, 0.00, 0.30) * 25.0
    if np.isfinite(earn_g):
        growth_score += norm(earn_g, 0.00, 0.35) * 25.0
    growth_score = float(np.clip(growth_score, 0.0, 100.0))

    quality_score = 50.0
    if np.isfinite(roe):
        quality_score += norm(roe, 0.08, 0.25) * 25.0
    if np.isfinite(roa):
        quality_score += norm(roa, 0.03, 0.12) * 15.0
    if np.isfinite(profit_m):
        quality_score += norm(profit_m, 0.05, 0.25) * 10.0
    if np.isfinite(op_m):
        quality_score += norm(op_m, 0.05, 0.25) * 10.0
    if np.isfinite(gross_m):
        quality_score += norm(gross_m, 0.20, 0.55) * 5.0
    quality_score = float(np.clip(quality_score, 0.0, 100.0))

    health_score = 50.0
    if np.isfinite(cr):
        health_score += norm(cr, 1.0, 3.0) * 25.0
    if np.isfinite(dte):
        health_score += norm(dte, 150.0, 20.0, invert=True) * 25.0
    if np.isfinite(ocf):
        health_score += 5.0 if ocf > 0 else -5.0
    if np.isfinite(fcf):
        health_score += 5.0 if fcf > 0 else -5.0
    health_score = float(np.clip(health_score, 0.0, 100.0))

    valuation_score = 50.0
    if np.isfinite(peg):
        valuation_score += norm(peg, 0.8, 2.5, invert=True) * 35.0
    elif np.isfinite(trailing_pe) or np.isfinite(forward_pe):
        pe = forward_pe if np.isfinite(forward_pe) else trailing_pe
        valuation_score += norm(pe, 8.0, 25.0, invert=True) * 25.0
    valuation_score = float(np.clip(valuation_score, 0.0, 100.0))

    fundamental_score = float(np.clip((growth_score * 0.35) + (quality_score * 0.30) + (health_score * 0.20) + (valuation_score * 0.15), 0.0, 100.0))

    if fundamental_score >= 80:
        grade = "A"
    elif fundamental_score >= 67:
        grade = "B"
    elif fundamental_score >= 55:
        grade = "C"
    elif fundamental_score >= 40:
        grade = "D"
    else:
        grade = "E"

    snap.update(
        {
            "fundamental_score": fundamental_score,
            "growth_score": growth_score,
            "quality_score": quality_score,
            "health_score": health_score,
            "valuation_score": valuation_score,
            "fundamental_grade": grade,
        }
    )
    return snap



def _safe_float(v, default=np.nan):
    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        out = float(v)
        return out if np.isfinite(out) else default
    except Exception:
        return default




def format_growth_percent(v, decimals: int = 0) -> str:
    """Format a growth value that may come as 0.18 or 18.0 into 18%."""
    try:
        if v is None or pd.isna(v):
            return "n/a"
        x = float(v)
        if abs(x) <= 1.5:
            x *= 100.0
        return f"{x:.{decimals}f}%"
    except Exception:
        return "n/a"


def _ensure_technical_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d.empty:
        return d
    if "EMA20" not in d.columns:
        d["EMA20"] = ema(d["Close"], 20)
    if "EMA50" not in d.columns:
        d["EMA50"] = ema(d["Close"], 50)
    if "EMA200" not in d.columns:
        d["EMA200"] = ema(d["Close"], 200)
    if "RSI14" not in d.columns:
        d["RSI14"] = rsi(d["Close"], 14)
    if "MACD_HIST" not in d.columns or "MACD" not in d.columns or "MACD_SIGNAL" not in d.columns:
        d["MACD"], d["MACD_SIGNAL"], d["MACD_HIST"] = macd(d["Close"])
    if "ATR14" not in d.columns:
        d["ATR14"] = atr(d, 14)
    if "ADX14" not in d.columns:
        d["ADX14"] = adx(d, 14)
    if "VOL_SMA20" not in d.columns:
        d["VOL_SMA20"] = d["Volume"].rolling(20).mean()
    if "REL_VOL" not in d.columns:
        d["REL_VOL"] = d["Volume"] / d["VOL_SMA20"]
    if "OBV" not in d.columns:
        d["OBV"] = obv(d)
    if "OBV_SLOPE10" not in d.columns:
        d["OBV_SLOPE10"] = d["OBV"] - d["OBV"].shift(10)
    if "CMF20" not in d.columns:
        d["CMF20"] = chaikin_money_flow(d, 20)
    if "MFI14" not in d.columns:
        d["MFI14"] = money_flow_index(d, 14)
    if "STOCH_K" not in d.columns or "STOCH_D" not in d.columns:
        d["STOCH_K"], d["STOCH_D"] = stochastic_oscillator(d, 14, 3, 3)
    if "CCI20" not in d.columns:
        d["CCI20"] = cci(d, 20)
    if "ROC12" not in d.columns:
        d["ROC12"] = rate_of_change(d["Close"], 12)
    return d


def _score_bucket(value: float, lo: float, hi: float, invert: bool = False) -> float:
    if value is None or pd.isna(value):
        return 50.0
    if hi == lo:
        return 50.0
    x = (float(value) - lo) / (hi - lo)
    x = float(np.clip(x, 0.0, 1.0))
    if invert:
        x = 1.0 - x
    return float(np.clip(x * 100.0, 0.0, 100.0))


def _trend_score(series: pd.Series) -> float:
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 3:
        return 50.0
    y = s.tail(min(8, len(s))).to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    try:
        slope = np.polyfit(x, y, 1)[0]
        scale = np.nanmean(np.abs(y)) + 1e-9
        return float(np.clip(50.0 + (slope / scale) * 250.0, 0.0, 100.0))
    except Exception:
        return 50.0


def compute_future_fundamental_grade(
    symbol: str,
    price_df: pd.DataFrame | None = None,
    macro_context: dict | None = None,
) -> dict:
    """Free-data proxy for forward fundamental quality.

    The main output should reflect the company's forward quality and trajectory.
    Macro context is kept as a separate risk overlay so a strong company is not
    mechanically downgraded into E just because the market regime is weak.
    """
    base = str(symbol).strip()
    snap = compute_fundamental_grade(base) if base else {}
    current_score = _safe_float(snap.get("fundamental_score"), 50.0)
    current_grade = str(snap.get("fundamental_grade", "n/a"))

    d = _ensure_technical_columns(price_df.copy()) if price_df is not None and not price_df.empty else pd.DataFrame()
    last = d.iloc[-1] if not d.empty else None

    # Current-fundamental quality is the anchor.
    current_block = float(np.clip(current_score, 0.0, 100.0))

    # Technical leading proxies that often front-run improved fundamentals.
    price_proxy = 50.0
    if last is not None:
        close = _safe_float(last.get("Close"))
        ema20 = _safe_float(last.get("EMA20"))
        ema50 = _safe_float(last.get("EMA50"))
        ema200 = _safe_float(last.get("EMA200"))
        rsi_v = _safe_float(last.get("RSI14"), 50.0)
        adx_v = _safe_float(last.get("ADX14"), 0.0)
        rel_vol_v = _safe_float(last.get("REL_VOL"), 1.0)
        cmf_v = _safe_float(last.get("CMF20"), 0.0)
        mfi_v = _safe_float(last.get("MFI14"), 50.0)
        obv_slope = _safe_float(last.get("OBV_SLOPE10"), 0.0)
        macd_hist = _safe_float(last.get("MACD_HIST"), 0.0)

        price_proxy = (
            float(close > ema20) * 14
            + float(ema20 > ema50) * 12
            + float(ema50 > ema200) * 10
            + float(rsi_v >= 50) * 8
            + float(adx_v >= 18) * 8
            + float(rel_vol_v >= 1.1) * 8
            + float(cmf_v > 0) * 12
            + float(mfi_v >= 50) * 6
            + float(obv_slope > 0) * 12
            + float(macd_hist > 0) * 10
        )
        price_proxy = float(np.clip(price_proxy, 0.0, 100.0))

    phase_info = classify_8_phase(d) if not d.empty and len(d) >= 60 else {"phase": "Unknown", "phase_confidence": 0.0}
    cycle_tuple = compute_cycle_features(d["Close"]) if not d.empty and len(d) >= 30 else (20, 999, False, {})
    dominant_period, time_to_bottom, cycle_ok, cycle_info = cycle_tuple if len(cycle_tuple) == 4 else (20, 999, False, {})
    time_to_top = _safe_float(cycle_info.get("time_to_next_top"), np.nan) if isinstance(cycle_info, dict) else np.nan
    cycle_reliability = _safe_float(cycle_info.get("cycle_reliability"), np.nan) if isinstance(cycle_info, dict) else np.nan

    # Macro regime is used as an overlay, not as the main company score.
    macro_multiplier = 1.0
    macro_gate_ok = True
    macro_gate_reason = "OK"
    macro_score = 50.0
    if isinstance(macro_context, dict) and macro_context:
        macro_multiplier = _safe_float(macro_context.get("macro_multiplier"), 1.0)
        macro_gate_ok = bool(macro_context.get("macro_gate_ok", True))
        macro_gate_reason = str(macro_context.get("macro_gate_reason", "OK"))
        macro_score = _safe_float(macro_context.get("macro_score"), 50.0)

    quality_score = current_block
    if quality_score < 35:
        quality_score = 35 + (quality_score * 0.4)

    growth_proxy = 50.0
    if not d.empty:
        try:
            rev_proxy = _score_bucket(d["Close"].pct_change(20).iloc[-1], -0.15, 0.25)
            mom_proxy = _score_bucket(d["Close"].pct_change(60).iloc[-1], -0.25, 0.40)
            accel_proxy = _score_bucket(d["Close"].pct_change(10).iloc[-1] - d["Close"].pct_change(30).iloc[-1], -0.20, 0.20)
            growth_proxy = float(np.clip((rev_proxy * 0.4) + (mom_proxy * 0.35) + (accel_proxy * 0.25), 0.0, 100.0))
        except Exception:
            growth_proxy = 50.0

    cash_flow_proxy = 50.0
    if not d.empty:
        cmf_v = _safe_float(last.get("CMF20"), 0.0) if last is not None else 0.0
        obv_slope = _safe_float(last.get("OBV_SLOPE10"), 0.0) if last is not None else 0.0
        rel_vol_v = _safe_float(last.get("REL_VOL"), 1.0) if last is not None else 1.0
        mfi_v = _safe_float(last.get("MFI14"), 50.0) if last is not None else 50.0
        cash_flow_proxy = float(np.clip(
            (50.0
             + (cmf_v * 60.0)
             + (np.clip(obv_slope / (abs(obv_slope) + 1e-9), -1, 1) * 8.0)
             + ((rel_vol_v - 1.0) * 18.0)
             + ((mfi_v - 50.0) * 0.6)),
            0.0,
            100.0
        ))

    balance_quality = 50.0
    cr = _safe_float(snap.get("current_ratio"), np.nan)
    dte = _safe_float(snap.get("debt_to_equity"), np.nan)
    roe = _safe_float(snap.get("return_on_equity"), np.nan)
    roa = _safe_float(snap.get("return_on_assets"), np.nan)
    op_margin = _safe_float(snap.get("operating_margin"), np.nan)
    gross_margin = _safe_float(snap.get("gross_margin"), np.nan)

    if np.isfinite(cr):
        balance_quality += _score_bucket(cr, 0.9, 3.0)
    if np.isfinite(dte):
        balance_quality += _score_bucket(dte, 20.0, 150.0, invert=True) * 0.8
    if np.isfinite(roe):
        balance_quality += _score_bucket(roe, 0.06, 0.25) * 0.8
    if np.isfinite(roa):
        balance_quality += _score_bucket(roa, 0.02, 0.10) * 0.5
    if np.isfinite(op_margin):
        balance_quality += _score_bucket(op_margin, 0.05, 0.25) * 0.7
    if np.isfinite(gross_margin):
        balance_quality += _score_bucket(gross_margin, 0.20, 0.55) * 0.5
    balance_quality = float(np.clip(balance_quality / 2.0, 0.0, 100.0))

    cycle_support = 50.0
    if isinstance(phase_info, dict):
        phase = str(phase_info.get("phase", "Unknown"))
        if phase in {"Early Accumulation", "Accumulation", "Late Accumulation"}:
            cycle_support += 15.0
        elif phase in {"Early Markup", "Markup"}:
            cycle_support += 10.0
        elif phase in {"Distribution", "Markdown"}:
            cycle_support -= 15.0

    if np.isfinite(cycle_reliability):
        cycle_support += float(np.clip((cycle_reliability - 50.0) * 0.35, -15.0, 15.0))
    if np.isfinite(time_to_bottom):
        cycle_support += float(np.clip((8.0 - time_to_bottom) * 1.6, -12.0, 12.0))
    if np.isfinite(time_to_top):
        cycle_support += float(np.clip((time_to_top - 6.0) * 0.8, -8.0, 8.0))
    cycle_support = float(np.clip(cycle_support, 0.0, 100.0))

    future_core_score = (
        current_block * 0.30
        + growth_proxy * 0.22
        + cash_flow_proxy * 0.18
        + balance_quality * 0.12
        + price_proxy * 0.10
        + cycle_support * 0.08
    )
    future_score = float(np.clip(future_core_score, 0.0, 100.0))
    future_macro_adjusted_score = float(np.clip(future_score * float(np.clip(macro_multiplier, 0.5, 1.15)), 0.0, 100.0))

    if future_score >= 80:
        grade = "A"
    elif future_score >= 67:
        grade = "B"
    elif future_score >= 55:
        grade = "C"
    elif future_score >= 40:
        grade = "D"
    else:
        grade = "E"

    if future_score - current_block >= 8:
        direction = "Improving"
    elif current_block - future_score >= 8:
        direction = "Deteriorating"
    else:
        direction = "Flat"

    confidence_source_count = 1
    if np.isfinite(price_proxy):
        confidence_source_count += 1
    if np.isfinite(cycle_reliability):
        confidence_source_count += 1
    if np.isfinite(balance_quality):
        confidence_source_count += 1
    confidence = float(np.clip(45 + confidence_source_count * 12 + (5 if macro_gate_ok else -8), 0.0, 100.0))

    return {
        "current_fundamental_score": current_block,
        "current_fundamental_grade": current_grade,
        "future_fundamental_score": future_score,
        "future_fundamental_grade": grade,
        "future_fundamental_direction": direction,
        "future_fundamental_confidence": confidence,
        "future_growth_proxy": growth_proxy,
        "future_cash_flow_proxy": cash_flow_proxy,
        "future_balance_quality": balance_quality,
        "future_price_proxy": price_proxy,
        "future_cycle_support": cycle_support,
        "future_macro_score": macro_score,
        "future_macro_adjusted_score": future_macro_adjusted_score,
        "future_macro_gate_ok": macro_gate_ok,
        "future_macro_gate_reason": macro_gate_reason,
        "future_moat_reason": f"{direction} | cycle={phase_info.get('phase', 'Unknown') if isinstance(phase_info, dict) else 'Unknown'}",
        "future_reliability": cycle_reliability,
        "future_time_to_top": time_to_top,
        "future_time_to_bottom": time_to_bottom,
        "future_phase": phase_info.get("phase", "Unknown") if isinstance(phase_info, dict) else "Unknown",
    }



def score_to_grade(score: float) -> str:
    try:
        s = float(score)
    except Exception:
        s = np.nan
    if not np.isfinite(s):
        return "n/a"
    if s >= 90:
        return "A+"
    if s >= 80:
        return "A"
    if s >= 70:
        return "B"
    if s >= 60:
        return "C"
    if s >= 50:
        return "D"
    return "E"


def compute_institutional_forward_score(
    symbol: str,
    price_df: pd.DataFrame | None = None,
    bench_df: pd.DataFrame | None = None,
    current_fundamental: dict | None = None,
    future_context: dict | None = None,
    technical_context: dict | None = None,
) -> dict:
    """Combine future fundamentals, accumulation, relative strength, quality, and catalyst into one score."""
    symbol = str(symbol).strip()
    current_fundamental = current_fundamental or {}
    future_context = future_context or {}
    technical_context = technical_context or {}

    price = price_df.copy() if price_df is not None else pd.DataFrame()
    if not price.empty:
        price = _ensure_technical_columns(price).dropna().copy()

    bench = bench_df.copy() if bench_df is not None else pd.DataFrame()
    if not bench.empty:
        bench = _ensure_technical_columns(bench).dropna().copy()

    last = price.iloc[-1] if not price.empty else None

    current_fundamental_score = _safe_float(current_fundamental.get("fundamental_score"), np.nan)
    if not np.isfinite(current_fundamental_score):
        current_fundamental_score = 50.0

    future_fundamental_score = _safe_float(future_context.get("future_fundamental_score"), np.nan)
    if not np.isfinite(future_fundamental_score):
        future_fundamental_score = current_fundamental_score

    future_confidence = _safe_float(future_context.get("future_fundamental_confidence"), 50.0)
    future_direction = str(future_context.get("future_fundamental_direction", "Flat"))
    future_phase = str(future_context.get("future_phase", "Unknown"))

    quality_score = float(np.clip(current_fundamental_score, 0.0, 100.0))

    smart_money_score = _safe_float(technical_context.get("smart_money_score"), np.nan)
    if not np.isfinite(smart_money_score):
        smart_money_score = 50.0

    cmf_score = 50.0
    obv_score = 50.0
    relvol_score = 50.0
    breakout_score = 50.0
    accel_score = 50.0
    phase_support = 50.0
    if last is not None:
        cmf_score = _score_bucket(_safe_float(last.get("CMF20"), 0.0), -0.15, 0.20)
        obv_slope = _safe_float(last.get("OBV_SLOPE10"), 0.0)
        obv_score = _score_bucket(obv_slope, -1e10, 1e10) if np.isfinite(obv_slope) else 50.0
        obv_score = 72.0 if obv_slope > 0 else 28.0 if obv_slope < 0 else 50.0
        relvol_score = _score_bucket(_safe_float(last.get("REL_VOL"), 1.0), 0.75, 1.75)
        close = _safe_float(last.get("Close"), np.nan)
        ema20 = _safe_float(last.get("EMA20"), np.nan)
        if np.isfinite(close) and np.isfinite(ema20) and ema20 != 0:
            breakout_score = _score_bucket((close / ema20) - 1.0, -0.06, 0.18)
        if len(price) >= 30:
            mom20 = price["Close"].pct_change(20).iloc[-1]
            mom60 = price["Close"].pct_change(60).iloc[-1]
            accel_score = _score_bucket(mom20 - mom60, -0.18, 0.22)
        phase_info = classify_8_phase(price) if len(price) >= 60 else {"phase": "Unknown", "phase_confidence": 0.0}
        phase = str(phase_info.get("phase", "Unknown"))
        if phase in {"Early Accumulation", "Accumulation", "Late Accumulation"}:
            phase_support = 82.0
        elif phase in {"Early Markup", "Markup"}:
            phase_support = 76.0
        elif phase in {"Late Markup"}:
            phase_support = 58.0
        elif phase in {"Distribution", "Markdown"}:
            phase_support = 26.0

    if not bench.empty and not price.empty:
        rs_line = compute_relative_strength(price["Close"], bench["Close"])
        if len(rs_line.dropna()) >= 3:
            rs63 = rs_line.pct_change(63).iloc[-1] if len(rs_line) > 63 else np.nan
            rs126 = rs_line.pct_change(126).iloc[-1] if len(rs_line) > 126 else np.nan
            rs252 = rs_line.pct_change(252).iloc[-1] if len(rs_line) > 252 else np.nan
            rs_components = [v for v in [rs63, rs126, rs252] if pd.notna(v)]
            if rs_components:
                rs_score = float(np.clip(
                    (
                        _score_bucket(rs63 if pd.notna(rs63) else np.nan, -0.20, 0.35) * 0.40
                        + _score_bucket(rs126 if pd.notna(rs126) else np.nan, -0.25, 0.50) * 0.35
                        + _score_bucket(rs252 if pd.notna(rs252) else np.nan, -0.30, 0.70) * 0.25
                    ),
                    0.0,
                    100.0,
                ))
            else:
                rs_score = 50.0
        else:
            rs_score = 50.0
    else:
        rs_score = 50.0

    accumulation_score = float(np.clip(
        (smart_money_score * 0.45)
        + (cmf_score * 0.18)
        + (obv_score * 0.16)
        + (relvol_score * 0.11)
        + (phase_support * 0.10),
        0.0,
        100.0,
    ))

    catalyst_score = float(np.clip(
        (future_confidence * 0.25)
        + (accel_score * 0.25)
        + (breakout_score * 0.20)
        + (phase_support * 0.20)
        + (60.0 if future_direction == "Improving" else 40.0 if future_direction == "Flat" else 25.0) * 0.10,
        0.0,
        100.0,
    ))

    ifs_score = float(np.clip(
        (future_fundamental_score * 0.30)
        + (accumulation_score * 0.25)
        + (rs_score * 0.20)
        + (quality_score * 0.15)
        + (catalyst_score * 0.10),
        0.0,
        100.0,
    ))

    return {
        "ifs_score": ifs_score,
        "ifs_grade": score_to_grade(ifs_score),
        "ifs_breakdown": {
            "Forward Fundamental": float(future_fundamental_score),
            "Accumulation": float(accumulation_score),
            "Relative Strength": float(rs_score),
            "Quality": float(quality_score),
            "Catalyst": float(catalyst_score),
        },
        "ifs_detail": {
            "future_direction": future_direction,
            "future_phase": future_phase,
            "future_confidence": float(future_confidence),
            "smart_money_score": float(smart_money_score),
            "quality_score": float(quality_score),
            "accumulation_score": float(accumulation_score),
            "relative_strength_score": float(rs_score),
            "catalyst_score": float(catalyst_score),
        },
    }


def parse_universe_text(text: str) -> list[str]:
    tokens: list[str] = []
    for line in text.splitlines():
        line = line.strip().upper()
        if not line:
            continue
        parts = [p.strip().upper() for p in line.replace(";", ",").split(",")]
        tokens.extend([p for p in parts if p])

    cleaned = []
    for t in tokens:
        norm = normalize_ticker(t)
        if norm:
            cleaned.append(norm)
    return list(dict.fromkeys(cleaned))


def load_universe_from_csv(source) -> list[str]:
    if source is None:
        return []
    try:
        dfu = pd.read_csv(source)
    except Exception:
        return []

    if dfu.empty:
        return []

    ticker_col = next(
        (
            col
            for col in dfu.columns
            if str(col).strip().lower() in {"ticker", "symbol", "kode", "code", "stock", "saham"}
        ),
        dfu.columns[0],
    )

    vals = dfu[ticker_col].astype(str).str.upper().str.strip().tolist()
    out = []
    for v in vals:
        norm = normalize_ticker(v)
        if norm:
            out.append(norm)
    return list(dict.fromkeys(out))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())



def compute_cycle_features(close: pd.Series) -> tuple[int, int, bool, dict]:
    close = close.dropna()
    n = len(close)
    if n < 30:
        return 20, 999, False, {
            "fft_period": np.nan,
            "hilbert_period": np.nan,
            "autocorr_period": np.nan,
            "weighted_period": 20,
            "fft_confidence": 0.0,
            "hilbert_confidence": 0.0,
            "autocorr_confidence": 0.0,
            "composite_confidence": 0.0,
            "cycle_reliability": 0.0,
            "anchor_idx": np.nan,
            "bars_since_anchor": np.nan,
            "time_to_next_top": np.nan,
            "phase_age_bars": np.nan,
            "phase_age_pct": np.nan,
            "time_to_next_bottom": 999,
            "threshold": np.nan,
            "cycle_position_pct": np.nan,
            "detrend_method": "HighPass+TailHilbert",
            "trend_lag_bars": np.nan,
            "cycle_gate_reason": "",
        }

    series = close.astype(float).copy().ffill().bfill()
    log_close = np.log(series.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    basis = log_close if len(log_close) >= 30 else series
    basis = basis.astype(float).copy().ffill().bfill()
    n_basis = len(basis)

    min_period = 5
    max_period = int(min(120, max(20, n_basis // 2)))
    max_period = max(min_period + 1, max_period)

    # Use a recent cycle window to keep the estimate responsive while still robust.
    cycle_window = int(np.clip(n_basis, 64, 160))
    cycle_arr = basis.to_numpy(dtype=float)[-cycle_window:]
    cycle_arr = cycle_arr - np.nanmean(cycle_arr)
    cycle_arr = np.nan_to_num(cycle_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Low-lag detrending: high-pass filter removes macro trend without center=True leakage.
    hp_period = int(np.clip(n_basis // 3, 20, 60))
    hp_series = highpass_filter(basis, hp_period)
    detrended = hp_series.dropna().astype(float).to_numpy(dtype=float)
    if detrended.size < 20:
        detrended = cycle_arr.copy()
    detrended = detrended - np.nanmean(detrended)
    detrended = np.nan_to_num(detrended, nan=0.0, posinf=0.0, neginf=0.0)

    def confidence_from_peak(peak: float, baseline: float) -> float:
        if not np.isfinite(peak):
            return 0.0
        base = abs(baseline) + 1e-9
        return float(np.clip(peak / base, 0.0, 10.0))

    fft_period = np.nan
    fft_conf = 0.0
    frequencies, power = periodogram(detrended)
    valid = (frequencies > 0) & (1 / frequencies >= min_period) & (1 / frequencies <= max_period)
    if np.any(valid):
        vf = frequencies[valid]
        vp = power[valid]
        if len(vp) > 0 and np.any(vp > 0):
            best_idx = int(np.argmax(vp))
            fft_freq = float(vf[best_idx])
            if fft_freq > 0:
                fft_period = float(np.clip(round(1 / fft_freq), min_period, max_period))
                fft_conf = confidence_from_peak(float(vp[best_idx]), float(np.median(vp) + 1e-9))

    hilbert_period = np.nan
    hilbert_conf = 0.0
    try:
        hilbert_window = int(np.clip(len(detrended), 64, 160))
        segment = detrended[-hilbert_window:] if len(detrended) > hilbert_window else detrended
        # Forecast-padding on the right tail reduces Hilbert edge distortion on the last bar.
        fit_points = min(20, len(segment))
        pad_future = max(8, min(16, max(8, len(segment) // 8)))
        segment_ext = linear_forecast_pad(segment, n_future=pad_future, fit_points=fit_points)
        analytic = hilbert(segment_ext[: len(segment) + pad_future])
        phase = np.unwrap(np.angle(analytic[: len(segment)]))
        dphase = np.diff(phase)
        if len(dphase) > 0:
            freq_series = np.abs(dphase) / (2 * np.pi)
            freq_series = freq_series[np.isfinite(freq_series) & (freq_series > 0)]
            if len(freq_series) > 0:
                median_freq = float(np.median(freq_series))
                if median_freq > 0:
                    hilbert_period = float(np.clip(round(1 / median_freq), min_period, max_period))
                    hilbert_conf = float(np.clip(1 - np.std(freq_series) / (np.mean(freq_series) + 1e-9), 0.0, 1.0))
    except Exception:
        hilbert_period = np.nan
        hilbert_conf = 0.0

    autocorr_period = np.nan
    autocorr_conf = 0.0
    x = detrended - np.mean(detrended)
    x_std = np.std(x)
    if x_std > 0:
        x = x / x_std
        corr_vals = []
        for lag in range(min_period, max_period + 1):
            if len(x) <= lag + 2:
                break
            c = np.corrcoef(x[:-lag], x[lag:])[0, 1]
            if np.isfinite(c):
                corr_vals.append((lag, c))
        if corr_vals:
            lag_arr = np.array([v[0] for v in corr_vals], dtype=float)
            c_arr = np.array([v[1] for v in corr_vals], dtype=float)
            valid_corr = c_arr > 0
            if np.any(valid_corr):
                best_idx = int(np.argmax(c_arr * valid_corr))
                autocorr_period = float(np.clip(lag_arr[best_idx], min_period, max_period))
                autocorr_conf = float(np.clip(c_arr[best_idx], 0.0, 1.0))

    candidates = []
    weights = []
    if np.isfinite(fft_period):
        candidates.append(float(fft_period))
        weights.append(float(np.clip(fft_conf, 0.1, 5.0)))
    if np.isfinite(hilbert_period):
        candidates.append(float(hilbert_period))
        weights.append(float(np.clip(hilbert_conf * 3.0, 0.1, 3.5)))
    if np.isfinite(autocorr_period):
        candidates.append(float(autocorr_period))
        weights.append(float(np.clip(autocorr_conf * 4.0, 0.1, 4.0)))

    if not candidates:
        dominant_period = int(np.clip(20, min_period, max_period))
    else:
        weights_arr = np.array(weights, dtype=float)
        candidate_arr = np.array(candidates, dtype=float)
        weighted_period = float(np.average(candidate_arr, weights=weights_arr))
        dominant_period = int(np.clip(round(weighted_period), min_period, max_period))

    order = int(np.clip(n_basis // 30, 2, 10))
    minima = argrelextrema(series.values, np.less_equal, order=order)[0]
    if len(minima) > 0:
        recent_cutoff = max(0, n_basis - max_period * 2)
        recent_minima = minima[minima >= recent_cutoff]
        anchor_idx = int(recent_minima[-1] if len(recent_minima) else minima[-1])
    else:
        window = min(max_period, n_basis)
        anchor_idx = int(max(0, n_basis - window))

    bars_since_anchor = max(0, (n_basis - 1) - anchor_idx)
    rem = bars_since_anchor % dominant_period
    time_to_next_bottom = 0 if rem == 0 else dominant_period - rem
    half_cycle = max(1, int(round(dominant_period / 2.0)))
    time_to_next_top = (half_cycle - rem) % dominant_period
    phase_age_bars = bars_since_anchor
    phase_age_pct = float(np.clip((phase_age_bars / max(dominant_period, 1)) * 100.0, 0.0, 100.0))
    threshold = max(4, int(round(dominant_period * 0.15)))

    composite_conf = float(np.clip(np.nanmean([fft_conf / 3.0, hilbert_conf, autocorr_conf]), 0.0, 1.0) * 100)
    if len(candidates) >= 2:
        candidate_arr = np.array(candidates, dtype=float)
        spread = float(np.std(candidate_arr) / (np.mean(candidate_arr) + 1e-9))
        agreement_score = float(np.clip(100.0 * (1.0 - spread), 0.0, 100.0))
    elif len(candidates) == 1:
        agreement_score = 60.0
    else:
        agreement_score = 0.0

    reliability = float(np.clip((composite_conf * 0.55) + (agreement_score * 0.45), 0.0, 100.0))
    cycle_ok = ((time_to_next_bottom <= threshold) or (bars_since_anchor <= threshold)) and (reliability >= 45.0)

    details = {
        "fft_period": int(fft_period) if np.isfinite(fft_period) else np.nan,
        "hilbert_period": int(hilbert_period) if np.isfinite(hilbert_period) else np.nan,
        "autocorr_period": int(autocorr_period) if np.isfinite(autocorr_period) else np.nan,
        "weighted_period": int(dominant_period),
        "fft_confidence": float(np.clip(fft_conf * 10, 0.0, 100.0)),
        "hilbert_confidence": float(np.clip(hilbert_conf * 100, 0.0, 100.0)),
        "autocorr_confidence": float(np.clip(autocorr_conf * 100, 0.0, 100.0)),
        "composite_confidence": composite_conf,
        "cycle_reliability": reliability,
        "anchor_idx": int(anchor_idx),
        "bars_since_anchor": int(bars_since_anchor),
        "threshold": int(threshold),
        "time_to_next_bottom": int(time_to_next_bottom),
        "time_to_next_top": int(time_to_next_top),
        "phase_age_bars": int(phase_age_bars),
        "phase_age_pct": float(phase_age_pct),
        "cycle_position_pct": float(np.clip((rem / max(dominant_period, 1)) * 100.0, 0.0, 100.0)),
        "detrend_method": "HighPass+TailHilbert",
        "trend_lag_bars": int(max(1, hp_period // 2)),
        "cycle_gate_reason": "",
        "cycle_window": int(cycle_window),
        "hilbert_window": int(min(len(detrended), 160)),
        "pad_future": int(max(8, min(16, max(8, len(segment) // 8)))) if 'segment' in locals() else 0,
    }

    return dominant_period, int(time_to_next_bottom), cycle_ok, details



def build_macro_liquidity_gate(bench_df: pd.DataFrame, benchmark_symbol: str = "^JKSE") -> dict:
    neutral = {
        "benchmark_symbol": benchmark_symbol,
        "macro_phase": "Unknown",
        "macro_phase_confidence": 0.0,
        "macro_period": np.nan,
        "macro_time_to_bottom": np.nan,
        "macro_time_to_top": np.nan,
        "macro_phase_age_bars": np.nan,
        "macro_phase_age_pct": np.nan,
        "macro_cycle_reliability": 0.0,
        "macro_cycle_gate_reason": "No benchmark data",
        "macro_score": 50.0,
        "macro_gate_ok": True,
        "macro_gate_reason": "OK",
        "macro_multiplier": 1.0,
        "cycle_tuple": (20, 999, False, {}),
        "benchmark_df": bench_df.copy() if bench_df is not None else pd.DataFrame(),
    }

    if bench_df is None or bench_df.empty:
        return neutral

    d = bench_df.copy()
    if d.empty or len(d) < 60:
        neutral["macro_cycle_gate_reason"] = "Benchmark data insufficient"
        neutral["benchmark_df"] = d
        return neutral

    d["EMA20"] = ema(d["Close"], 20)
    d["EMA50"] = ema(d["Close"], 50)
    d["EMA200"] = ema(d["Close"], 200)
    d["RSI14"] = rsi(d["Close"], 14)
    d["ATR14"] = atr(d, 14)
    d["ADX14"] = adx(d, 14)
    d["VOL_SMA20"] = d["Volume"].rolling(20).mean()
    d["REL_VOL"] = d["Volume"] / d["VOL_SMA20"]
    d["OBV"] = obv(d)
    d["OBV_SLOPE10"] = d["OBV"] - d["OBV"].shift(10)
    d["CMF20"] = chaikin_money_flow(d, 20)
    d["MFI14"] = money_flow_index(d, 14)
    d["STOCH_K"], d["STOCH_D"] = stochastic_oscillator(d, 14, 3, 3)
    d["CCI20"] = cci(d, 20)
    d["ROC12"] = rate_of_change(d["Close"], 12)
    d = d.dropna().copy()

    if d.empty or len(d) < 60:
        neutral["macro_cycle_gate_reason"] = "Benchmark data insufficient after indicators"
        neutral["benchmark_df"] = d
        return neutral

    last = d.iloc[-1]
    cycle_tuple = compute_cycle_features(d["Close"])
    phase_info = classify_8_phase(d)

    dominant_period, time_to_bottom, _, cycle_info = cycle_tuple
    adx_last = float(last["ADX14"]) if pd.notna(last["ADX14"]) else np.nan
    cycle_reliability = float(cycle_info.get("cycle_reliability", np.nan)) if pd.notna(cycle_info.get("cycle_reliability", np.nan)) else np.nan

    macro_phase = str(phase_info.get("phase", "Unknown"))
    macro_phase_confidence = float(phase_info.get("phase_confidence", 0.0))
    macro_time_to_top = int(cycle_info.get("time_to_next_top", np.nan)) if pd.notna(cycle_info.get("time_to_next_top", np.nan)) else np.nan
    macro_phase_age_bars = int(cycle_info.get("phase_age_bars", np.nan)) if pd.notna(cycle_info.get("phase_age_bars", np.nan)) else np.nan
    macro_phase_age_pct = float(cycle_info.get("phase_age_pct", np.nan)) if pd.notna(cycle_info.get("phase_age_pct", np.nan)) else np.nan

    macro_score = 100.0
    reasons = []

    if macro_phase == "Markdown":
        macro_score -= 40.0
        reasons.append("IHSG phase Markdown")
    elif macro_phase == "Distribution":
        macro_score -= 22.0
        reasons.append("IHSG phase Distribution")
    elif macro_phase == "Late Markup":
        macro_score -= 10.0

    if np.isfinite(adx_last) and adx_last > 35:
        macro_score -= 25.0
        reasons.append(f"IHSG ADX {adx_last:.0f} > 35")
    elif np.isfinite(adx_last) and adx_last > 28:
        macro_score -= 12.0
        reasons.append(f"IHSG ADX {adx_last:.0f} elevated")

    if np.isfinite(cycle_reliability) and cycle_reliability < 45:
        macro_score -= 15.0
        reasons.append(f"IHSG CycleRel {cycle_reliability:.0f} < 45")
    elif np.isfinite(cycle_reliability) and cycle_reliability < 60:
        macro_score -= 8.0
        reasons.append(f"IHSG CycleRel {cycle_reliability:.0f} moderate")

    if np.isfinite(last["REL_VOL"]) and float(last["REL_VOL"]) < 0.8:
        macro_score -= 5.0
        reasons.append("IHSG relative volume weak")

    macro_score = float(np.clip(macro_score, 0.0, 100.0))
    macro_gate_ok = (macro_phase != "Markdown") and (not (np.isfinite(adx_last) and adx_last > 35)) and (not (np.isfinite(cycle_reliability) and cycle_reliability < 45)) and (macro_score >= 55.0)
    macro_gate_reason = "OK" if macro_gate_ok else ", ".join(reasons) if reasons else "Macro gate off"
    macro_multiplier = 1.0 if macro_gate_ok else (0.72 if macro_score >= 40 else 0.55)

    return {
        "benchmark_symbol": benchmark_symbol,
        "macro_phase": macro_phase,
        "macro_phase_confidence": macro_phase_confidence,
        "macro_period": int(dominant_period) if np.isfinite(dominant_period) else np.nan,
        "macro_time_to_bottom": int(time_to_bottom) if np.isfinite(time_to_bottom) else np.nan,
        "macro_time_to_top": macro_time_to_top,
        "macro_phase_age_bars": macro_phase_age_bars,
        "macro_phase_age_pct": macro_phase_age_pct,
        "macro_cycle_reliability": cycle_reliability if np.isfinite(cycle_reliability) else np.nan,
        "macro_cycle_gate_reason": cycle_info.get("cycle_gate_reason", "OK"),
        "macro_score": macro_score,
        "macro_gate_ok": macro_gate_ok,
        "macro_gate_reason": macro_gate_reason,
        "macro_multiplier": macro_multiplier,
        "cycle_tuple": cycle_tuple,
        "benchmark_df": d,
        "benchmark_last": last,
        "benchmark_adx": adx_last,
        "benchmark_cycle_info": cycle_info,
    }


def compute_relative_strength(stock_close: pd.Series, bench_close: pd.Series) -> pd.Series:
    aligned = pd.concat([stock_close.rename("stock"), bench_close.rename("bench")], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned["stock"] / aligned["bench"]


def classify_8_phase(d: pd.DataFrame) -> dict:
    x = d.dropna().copy()
    if x.empty or len(x) < 60:
        return {
            "phase": "Unknown",
            "phase_confidence": 0.0,
            "phase_rank": 0.0,
            "phase_reason": "Data historis belum cukup untuk klasifikasi phase.",
            "phase_scores": {},
        }

    last = x.iloc[-1]
    recent = x.tail(min(120, len(x))).copy()

    high20 = float(recent["High"].tail(20).max())
    low20 = float(recent["Low"].tail(20).min())
    high60 = float(recent["High"].max())
    low60 = float(recent["Low"].min())

    def safe_div(a, b):
        return float(a / b) if np.isfinite(b) and b != 0 else np.nan

    pos20 = safe_div(float(last["Close"]) - low20, high20 - low20)
    pos60 = safe_div(float(last["Close"]) - low60, high60 - low60)
    pos20 = float(np.clip(pos20 if np.isfinite(pos20) else 0.5, 0.0, 1.0))
    pos60 = float(np.clip(pos60 if np.isfinite(pos60) else 0.5, 0.0, 1.0))

    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    ema200 = float(last["EMA200"])
    close = float(last["Close"])
    rsi_v = float(last["RSI14"]) if pd.notna(last["RSI14"]) else 50.0
    adx_v = float(last["ADX14"]) if pd.notna(last["ADX14"]) else 0.0
    rel_vol_v = float(last["REL_VOL"]) if pd.notna(last["REL_VOL"]) else 1.0
    cmf_v = float(last["CMF20"]) if "CMF20" in x.columns and pd.notna(last["CMF20"]) else 0.0
    mfi_v = float(last["MFI14"]) if "MFI14" in x.columns and pd.notna(last["MFI14"]) else 50.0
    stoch_k_v = float(last["STOCH_K"]) if "STOCH_K" in x.columns and pd.notna(last["STOCH_K"]) else 50.0
    stoch_d_v = float(last["STOCH_D"]) if "STOCH_D" in x.columns and pd.notna(last["STOCH_D"]) else 50.0
    cci_v = float(last["CCI20"]) if "CCI20" in x.columns and pd.notna(last["CCI20"]) else 0.0
    roc_v = float(last["ROC12"]) if "ROC12" in x.columns and pd.notna(last["ROC12"]) else 0.0
    obv_slope = float(last["OBV_SLOPE10"]) if pd.notna(last["OBV_SLOPE10"]) else 0.0

    ema20_slope = float(last["EMA20"] - x["EMA20"].iloc[max(0, len(x) - 6)]) if len(x) >= 6 else 0.0
    atr14 = float(last["ATR14"]) if pd.notna(last["ATR14"]) else max(close * 0.02, 1.0)

    bull_stack = (ema20 > ema50) and (ema50 > ema200)
    bear_stack = (ema20 < ema50) and (ema50 < ema200)
    above_ema20 = close > ema20
    above_ema50 = close > ema50
    above_ema200 = close > ema200
    breakout20 = close > high20 * 1.001
    breakdown20 = close < low20 * 0.999
    extended = (close - ema20) / atr14 if atr14 > 0 else 0.0

    low_regime = float(np.clip(1 - pos60, 0, 1))
    high_regime = float(np.clip(pos60, 0, 1))

    rsi_low = float(np.clip((55 - rsi_v) / 25, 0, 1))
    rsi_mid = float(np.clip(1 - abs(rsi_v - 60) / 18, 0, 1))
    rsi_very_low = float(np.clip((45 - rsi_v) / 20, 0, 1))

    adx_low = float(np.clip((20 - adx_v) / 20, 0, 1))
    adx_mid = float(np.clip(1 - abs(adx_v - 24) / 12, 0, 1))
    adx_high = float(np.clip((adx_v - 18) / 20, 0, 1))

    obv_up_score = float(np.clip((obv_slope > 0) * 1.0, 0, 1))
    obv_down_score = float(np.clip((obv_slope < 0) * 1.0, 0, 1))
    ema_bull = float(np.clip((bull_stack) * 1.0, 0, 1))
    ema_bear = float(np.clip((bear_stack) * 1.0, 0, 1))

    range_width = float((high20 - low20) / close) if close > 0 else 0.0
    compression = float(np.clip(1 - range_width / 0.18, 0, 1))

    scores = {
        "Early Accumulation": (
            low_regime * 35
            + adx_low * 20
            + obv_up_score * 18
            + rsi_low * 12
            + float(ema20_slope >= 0) * 5
            + float(cmf_v > 0) * 8
            + float(mfi_v <= 55) * 6
            + float(not bear_stack) * 10
            + float(cmf_v > 0) * 6
            + float(stoch_k_v >= stoch_d_v) * 6
        ),
        "Accumulation": (
            compression * 25
            + float(np.clip(1 - abs(pos60 - 0.35) / 0.25, 0, 1)) * 20
            + adx_low * 15
            + obv_up_score * 20
            + rsi_mid * 10
            + float(not bear_stack) * 10
        ),
        "Late Accumulation": (
            float(np.clip(1 - abs(pos60 - 0.55) / 0.25, 0, 1)) * 18
            + float(breakout20 or above_ema50) * 25
            + obv_up_score * 18
            + float(ema20 > ema50 or ema20_slope > 0) * 15
            + float(50 <= rsi_v <= 65) * 10
            + float(stoch_k_v >= stoch_d_v) * 8
            + adx_mid * 12
        ),
        "Early Markup": (
            float(breakout20) * 22
            + float(above_ema20 and above_ema50) * 20
            + float(ema20 > ema50) * 18
            + float(ema50 >= ema200) * 8
            + obv_up_score * 15
            + float(52 <= rsi_v <= 68) * 8
            + float(cmf_v > 0) * 6
            + adx_high * 7
        ),
        "Markup": (
            ema_bull * 28
            + float(above_ema20 and above_ema50 and above_ema200) * 15
            + obv_up_score * 18
            + float(55 <= rsi_v <= 75) * 15
            + float(stoch_k_v >= stoch_d_v) * 6
            + adx_high * 16
            + high_regime * 8
        ),
        "Late Markup": (
            ema_bull * 22
            + high_regime * 18
            + float(rsi_v >= 70) * 18
            + float(extended > 1.0) * 14
            + float(mfi_v >= 70) * 8
            + float(adx_v >= 20) * 10
            + obv_down_score * 8
            + float((rel_vol_v < 1.0) or (obv_slope <= 0)) * 10
        ),
        "Distribution": (
            high_regime * 24
            + float(rsi_v >= 60) * 10
            + obv_down_score * 20
            + float((close < ema20) or (close < ema50)) * 18
            + float((not breakout20) and (close < high20 * 0.995)) * 14
            + float(ema20_slope <= 0) * 8
            + float((adx_v >= 18) and (adx_v <= 30)) * 6
            + float(cmf_v < 0) * 8
        ),
        "Markdown": (
            ema_bear * 28
            + float(breakdown20) * 20
            + rsi_very_low * 16
            + obv_down_score * 18
            + float(close < ema50) * 10
            + float(pos60 < 0.45) * 8
            + float(adx_v >= 18) * 6
        ),
    }

    phase = max(scores, key=scores.get)
    sorted_scores = sorted(scores.values(), reverse=True)
    best = float(sorted_scores[0])
    second = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
    confidence = float(np.clip((best - second) + 50, 0, 100))

    reasons = {
        "Early Accumulation": "Harga masih dekat area bawah, OBV mulai membaik, momentum lemah namun stabil.",
        "Accumulation": "Base sedang terbentuk, volatilitas terkompresi, akumulasi relatif dominan.",
        "Late Accumulation": "Harga mulai keluar dari base dan bersiap transisi ke markup.",
        "Early Markup": "Breakout awal dan struktur mulai bullish, namun belum sepenuhnya matang.",
        "Markup": "Struktur bullish sudah jelas, momentum dan trend stack mendukung kelanjutan tren.",
        "Late Markup": "Tren masih naik tetapi sudah extended dan mulai rawan distribusi.",
        "Distribution": "Harga tinggi tetapi momentum melemah, tanda selling into strength mulai muncul.",
        "Markdown": "Struktur bearish dominan, tekanan jual menguasai.",
    }

    return {
        "phase": phase,
        "phase_confidence": confidence,
        "phase_rank": best,
        "phase_reason": reasons.get(phase, "-"),
        "phase_scores": scores,
    }


def detect_reversal_signals(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()

    x["Bullish_Engulfing"] = (
        (x["Close"] > x["Open"])
        & (x["Close"].shift(1) < x["Open"].shift(1))
        & (x["Close"] >= x["Open"].shift(1))
        & (x["Open"] <= x["Close"].shift(1))
    )

    body = (x["Close"] - x["Open"]).abs()
    candle_range = (x["High"] - x["Low"]).replace(0, np.nan)
    lower_wick = np.minimum(x["Open"], x["Close"]) - x["Low"]
    upper_wick = x["High"] - np.maximum(x["Open"], x["Close"])

    x["Hammer"] = (body / candle_range <= 0.35) & (lower_wick >= body * 2) & (upper_wick <= body)
    x["Inverted_Hammer"] = (body / candle_range <= 0.35) & (upper_wick >= body * 2) & (lower_wick <= body)

    prev2_bear = x["Close"].shift(2) < x["Open"].shift(2)
    prev1_small = (x["Close"].shift(1) - x["Open"].shift(1)).abs() <= (x["High"].shift(1) - x["Low"].shift(1)) * 0.35
    curr_bull = x["Close"] > x["Open"]
    x["Morning_Star"] = prev2_bear & prev1_small & curr_bull & (x["Close"] > (x["Open"].shift(2) + x["Close"].shift(2)) / 2)

    x["EMA20_Reclaim"] = (x["Close"] > x["EMA20"]) & (x["Close"].shift(1) <= x["EMA20"].shift(1))
    x["MACD_Bull_Cross"] = (x["MACD"] > x["MACD_SIGNAL"]) & (x["MACD"].shift(1) <= x["MACD_SIGNAL"].shift(1))
    x["RSI_Bounce"] = (x["RSI14"] > 50) & (x["RSI14"].shift(1) <= 50)
    x["Breakout_5D"] = x["Close"] > x["High"].rolling(5).max().shift(1)
    x["Volume_Surge"] = x["REL_VOL"] >= 1.25
    x["Bullish_FVG"] = (x["Low"] > x["High"].shift(2)) & (x["Close"].shift(1) > x["Open"].shift(1))
    x["Bullish_OB"] = (x["Close"].shift(1) < x["Open"].shift(1)) & x["Breakout_5D"]

    return x


def map_flow_to_score(flow_mode: str) -> float:
    return make_flow_score(flow_mode)


# =========================================================
# Safety fallbacks (no-op when helpers already exist)
# =========================================================
if "parse_universe_text" not in globals():
    def parse_universe_text(text: str) -> list[str]:
        tokens: list[str] = []
        for line in str(text).splitlines():
            line = line.strip().upper()
            if not line:
                continue
            parts = [p.strip().upper() for p in line.replace(";", ",").split(",")]
            tokens.extend([p for p in parts if p])

        cleaned = []
        for t in tokens:
            norm = normalize_ticker(t)
            if norm:
                cleaned.append(norm)
        return list(dict.fromkeys(cleaned))

if "load_universe_from_csv" not in globals():
    def load_universe_from_csv(source) -> list[str]:
        if source is None:
            return []
        try:
            dfu = pd.read_csv(source)
        except Exception:
            return []

        if dfu.empty:
            return []

        ticker_col = next(
            (
                col
                for col in dfu.columns
                if str(col).strip().lower() in {"ticker", "symbol", "kode", "code", "stock", "saham"}
            ),
            dfu.columns[0],
        )

        vals = dfu[ticker_col].astype(str).str.upper().str.strip().tolist()
        out = []
        for v in vals:
            norm = normalize_ticker(v)
            if norm:
                out.append(norm)
        return list(dict.fromkeys(out))

if "map_flow_to_score" not in globals():
    def map_flow_to_score(flow_mode: str) -> float:
        return make_flow_score(flow_mode)

if "build_macro_liquidity_gate" not in globals():
    def build_macro_liquidity_gate(bench_df: pd.DataFrame, benchmark_symbol: str = "^JKSE") -> dict:
        return {
            "benchmark_symbol": benchmark_symbol,
            "macro_phase": "Unknown",
            "macro_phase_confidence": 0.0,
            "macro_period": np.nan,
            "macro_time_to_bottom": np.nan,
            "macro_time_to_top": np.nan,
            "macro_phase_age_bars": np.nan,
            "macro_phase_age_pct": np.nan,
            "macro_cycle_reliability": 0.0,
            "macro_cycle_gate_reason": "Macro gate unavailable",
            "macro_score": 50.0,
            "macro_gate_ok": True,
            "macro_gate_reason": "OK",
            "macro_multiplier": 1.0,
            "cycle_tuple": (20, 999, False, {}),
            "benchmark_df": bench_df.copy() if bench_df is not None else pd.DataFrame(),
        }


# =========================================================
# Scoring Engine
# =========================================================
def score_stock_smc(
    df: pd.DataFrame,
    flow_used: bool,
    flow_val: float,
    min_avg_volume: float,
    min_price: float,
    max_price: float,
    mode: str,
    min_history_bars: int,
    macro_context: dict | None = None,
    future_fundamental_context: dict | None = None,
) -> dict:
    d = df.copy()
    if d.empty or len(d) < min_history_bars:
        return {"valid": False, "reason": "Data historis tidak mencukupi"}

    d["EMA20"] = ema(d["Close"], 20)
    d["EMA50"] = ema(d["Close"], 50)
    d["EMA200"] = ema(d["Close"], 200)
    d["RSI14"] = rsi(d["Close"], 14)
    d["MACD"], d["MACD_SIGNAL"], d["MACD_HIST"] = macd(d["Close"])
    d["ATR14"] = atr(d, 14)
    d["ADX14"] = adx(d, 14)
    d["BB_MID"], d["BB_UPPER"], d["BB_LOWER"] = bollinger(d["Close"], 20, 2.0)
    d["VOL_SMA20"] = d["Volume"].rolling(20).mean()
    d["REL_VOL"] = d["Volume"] / d["VOL_SMA20"]
    d["VPT"] = (d["Volume"] * d["Close"].pct_change()).cumsum()
    d["OBV"] = obv(d)
    d["OBV_SMA10"] = d["OBV"].rolling(10).mean()
    d["OBV_SLOPE10"] = d["OBV"] - d["OBV"].shift(10)
    d["CMF20"] = chaikin_money_flow(d, 20)
    d["MFI14"] = money_flow_index(d, 14)
    d["STOCH_K"], d["STOCH_D"] = stochastic_oscillator(d, 14, 3, 3)
    d["CCI20"] = cci(d, 20)
    d["ROC12"] = rate_of_change(d["Close"], 12)

    # OBV slope is used later in scoring, so define it before any score calculations.
    obv_slope = float(d["OBV_SLOPE10"].iloc[-1]) if len(d) > 0 and pd.notna(d["OBV_SLOPE10"].iloc[-1]) else 0.0

    d = detect_reversal_signals(d)
    d = d.dropna().copy()
    if len(d) < max(50, int(min_history_bars * 0.7)):
        return {"valid": False, "reason": "Kebocoran data setelah dropping NaN"}

    last = d.iloc[-1]
    prev = d.iloc[-2]
    dominant_period, time_to_next_bottom, cycle_ok, cycle_info = compute_cycle_features(d["Close"])
    phase_info = classify_8_phase(d)

    adx_last = float(last["ADX14"]) if pd.notna(last["ADX14"]) else np.nan
    time_to_next_top = int(cycle_info.get("time_to_next_top", max(1, dominant_period // 2)))
    phase_age_bars = int(cycle_info.get("phase_age_bars", 0)) if pd.notna(cycle_info.get("phase_age_bars", np.nan)) else np.nan
    phase_age_pct = float(cycle_info.get("phase_age_pct", np.nan)) if pd.notna(cycle_info.get("phase_age_pct", np.nan)) else np.nan
    cycle_reliability = float(cycle_info.get("cycle_reliability", np.nan)) if pd.notna(cycle_info.get("cycle_reliability", np.nan)) else np.nan

    cycle_gate_reason = []
    if phase_info.get("phase") == "Markdown":
        cycle_ok = False
        cycle_gate_reason.append("phase Markdown")
    if np.isfinite(adx_last) and adx_last > 35:
        cycle_ok = False
        cycle_gate_reason.append(f"ADX {adx_last:.0f} > 35")
    if np.isfinite(cycle_reliability) and cycle_reliability < 45:
        cycle_ok = False
        cycle_gate_reason.append(f"CycleRel {cycle_reliability:.0f} < 45")
    cycle_info["cycle_gate_reason"] = ", ".join(cycle_gate_reason) if cycle_gate_reason else "OK"

    macro_context = macro_context or {}
    macro_symbol = str(macro_context.get("benchmark_symbol", "^JKSE"))
    macro_phase = str(macro_context.get("macro_phase", "Unknown"))
    macro_score = float(macro_context.get("macro_score", np.nan)) if pd.notna(macro_context.get("macro_score", np.nan)) else np.nan
    macro_gate_ok = bool(macro_context.get("macro_gate_ok", True))
    macro_gate_reason = str(macro_context.get("macro_gate_reason", "OK"))
    macro_multiplier = float(macro_context.get("macro_multiplier", 1.0)) if pd.notna(macro_context.get("macro_multiplier", 1.0)) else 1.0
    macro_cycle_reliability = float(macro_context.get("macro_cycle_reliability", np.nan)) if pd.notna(macro_context.get("macro_cycle_reliability", np.nan)) else np.nan
    macro_time_to_bottom = macro_context.get("macro_time_to_bottom", np.nan)
    macro_time_to_top = macro_context.get("macro_time_to_top", np.nan)
    macro_phase_age_bars = macro_context.get("macro_phase_age_bars", np.nan)
    macro_phase_age_pct = macro_context.get("macro_phase_age_pct", np.nan)

    reversal_names = [
        "Bullish_Engulfing",
        "Hammer",
        "Inverted_Hammer",
        "Morning_Star",
        "EMA20_Reclaim",
        "MACD_Bull_Cross",
        "RSI_Bounce",
        "Breakout_5D",
    ]
    reversal_score = 0
    reversal_hits = []
    for name in reversal_names:
        if bool(d[name].tail(5).any()):
            reversal_score += 1
            reversal_hits.append(name)

    smc_points = 0
    smc_points += 4 * int(d["Bullish_FVG"].tail(5).any())
    smc_points += 4 * int(d["Bullish_OB"].tail(5).any())
    smc_points += 2 * int(float(last["REL_VOL"]) >= 1.25)
    smc_points += 2 * int(float(last["VPT"]) > float(d["VPT"].iloc[-5])) if len(d) >= 5 else 1

    trend_points = 0
    trend_points += int(last["Close"] > last["EMA20"])
    trend_points += int(last["EMA20"] > last["EMA50"])
    trend_points += int(last["EMA50"] > last["EMA200"])
    trend_points += int(last["EMA50"] > prev["EMA50"])

    momentum_points = 0
    momentum_points += int(50 <= float(last["RSI14"]) <= 72)
    momentum_points += int(float(last["MACD_HIST"]) > 0)
    momentum_points += int(last["Close"] > last["BB_MID"])
    momentum_points += int(float(last["ADX14"]) >= 18)

    reversal_points = 0
    reversal_points += int(d["EMA20_Reclaim"].tail(5).any())
    reversal_points += int(d["MACD_Bull_Cross"].tail(5).any())
    reversal_points += int(d["RSI_Bounce"].tail(5).any())
    reversal_points += int(d["Breakout_5D"].tail(5).any())

    cmf_last = float(last["CMF20"]) if pd.notna(last["CMF20"]) else 0.0
    mfi_last = float(last["MFI14"]) if pd.notna(last["MFI14"]) else 50.0
    stoch_k_last = float(last["STOCH_K"]) if pd.notna(last["STOCH_K"]) else 50.0
    stoch_d_last = float(last["STOCH_D"]) if pd.notna(last["STOCH_D"]) else 50.0

    smart_money_score = 0.0
    smart_money_score += 18.0 * float(d["Bullish_FVG"].tail(8).any())
    smart_money_score += 18.0 * float(d["Bullish_OB"].tail(8).any())
    smart_money_score += 12.0 * float(np.clip((float(last["REL_VOL"]) - 0.75) / 1.0, 0.0, 1.0))
    smart_money_score += 14.0 * float(np.clip((cmf_last + 0.5) / 1.0, 0.0, 1.0))
    smart_money_score += 10.0 * float(np.clip((mfi_last - 45.0) / 30.0, 0.0, 1.0))
    smart_money_score += 10.0 * float(np.clip(((stoch_k_last - stoch_d_last) + 15.0) / 30.0, 0.0, 1.0))
    smart_money_score += 10.0 * float(obv_slope > 0)
    smart_money_score += 8.0 * float(len(d) >= 5 and float(last["VPT"]) > float(d["VPT"].iloc[-5]))
    smart_money_score = float(np.clip(smart_money_score, 0.0, 100.0))

    core_raw = (smc_points * 4) + (trend_points * 3) + (momentum_points * 2) + (reversal_points * 3)
    core_max = (12 * 4) + (4 * 3) + (4 * 2) + (4 * 3)
    core_score = (core_raw / core_max) * 100 if core_max > 0 else 0.0

    final_score = ((core_score * 0.55) + (smart_money_score * 0.25) + (flow_val * 0.20)) if flow_used else ((core_score * 0.65) + (smart_money_score * 0.35))
    if np.isfinite(macro_multiplier):
        final_score *= macro_multiplier

    future_fundamental_score = np.nan
    future_fundamental_grade = "n/a"
    future_fundamental_direction = "n/a"
    future_fundamental_confidence = np.nan
    future_fundamental_phase = "Unknown"
    future_fundamental_reason = "n/a"
    if future_fundamental_context is not None:
        future_fundamental_score = _safe_float(future_fundamental_context.get("future_fundamental_score"), np.nan)
        future_fundamental_grade = str(future_fundamental_context.get("future_fundamental_grade", "n/a"))
        future_fundamental_direction = str(future_fundamental_context.get("future_fundamental_direction", "n/a"))
        future_fundamental_confidence = _safe_float(future_fundamental_context.get("future_fundamental_confidence"), np.nan)
        future_fundamental_phase = str(future_fundamental_context.get("future_phase", "Unknown"))
        future_fundamental_reason = str(future_fundamental_context.get("future_moat_reason", "n/a"))
        if np.isfinite(future_fundamental_score):
            final_score = float(np.clip((final_score * 0.78) + (future_fundamental_score * 0.22), 0.0, 100.0))

    liquidity_ok = (d["Volume"].tail(20).mean() >= min_avg_volume) and (min_price <= float(last["Close"]) <= max_price)
    trend_ok = (last["Close"] > last["EMA20"]) and (last["EMA50"] > last["EMA200"])
    smc_confirmed = d["Bullish_FVG"].tail(8).any() or d["Bullish_OB"].tail(8).any()

    if mode == "Conservative":
        buy_threshold, strong_threshold = 80, 88
    elif mode == "Balanced":
        buy_threshold, strong_threshold = 70, 83
    else:
        buy_threshold, strong_threshold = 60, 74

    if not macro_gate_ok:
        if liquidity_ok and trend_ok and smc_confirmed and (final_score >= buy_threshold):
            decision = "WATCHLIST"
        elif liquidity_ok and (final_score >= buy_threshold - 8):
            decision = "WATCHLIST"
        else:
            decision = "AVOID"
    else:
        if liquidity_ok and trend_ok and smc_confirmed and (final_score >= strong_threshold):
            decision = "STRONG BUY"
        elif liquidity_ok and trend_ok and (final_score >= buy_threshold):
            decision = "BUY"
        elif liquidity_ok and (final_score >= buy_threshold - 10):
            decision = "WATCHLIST"
        else:
            decision = "AVOID"

    recent_swing_low = float(d["Low"].tail(10).min())
    recent_support_ema = float(d["EMA20"].iloc[-1])
    ob_zone = np.nan
    ob_rows = d[d["Bullish_OB"]].tail(3)
    if not ob_rows.empty:
        ob_idx = ob_rows.index[-1]
        loc = d.index.get_loc(ob_idx)
        if loc >= 1:
            ob_zone = float((d["Low"].iloc[loc - 1] + d["High"].iloc[loc - 1]) / 2)

    if decision in {"BUY", "STRONG BUY"}:
        entry_candidates = [float(last["Close"]), recent_support_ema, recent_swing_low + float(last["ATR14"]) * 0.25]
        if np.isfinite(ob_zone):
            entry_candidates.append(ob_zone)
        entry_candidates = [v for v in entry_candidates if np.isfinite(v)]
        entry_price = float(np.nanmean(entry_candidates)) if entry_candidates else np.nan
        entry_price = max(entry_price, 0.0)
        stop_price = min(recent_swing_low, float(last["Close"]) - float(last["ATR14"]) * 1.0)
        stop_price = max(stop_price, 0.0)
    else:
        entry_price = np.nan
        stop_price = np.nan

    obv_slope = float(last["OBV_SLOPE10"]) if pd.notna(last["OBV_SLOPE10"]) else np.nan
    if pd.isna(obv_slope):
        obv_trend = "Flat"
    elif obv_slope > 0:
        obv_trend = "Rising"
    elif obv_slope < 0:
        obv_trend = "Falling"
    else:
        obv_trend = "Flat"

    notes = []
    if not liquidity_ok:
        notes.append("Filter_Likuiditas_Gagal")
    if not trend_ok:
        notes.append("Struktur_Trend_Bearish")
    if not smc_confirmed:
        notes.append("Tanpa_FVG/OB_Institusi")
    if not cycle_ok:
        notes.append("Siklus_Belum_Menguat")
        if cycle_gate_reason:
            notes.append("Cycle_Gated_" + "_".join(cycle_gate_reason).replace(" ", ""))
    if not macro_gate_ok:
        notes.append("Macro_Gated_" + macro_gate_reason.replace(" ", "_"))
    if reversal_score == 0:
        notes.append("Belum_Ada_Reversal_Strong")


    trend_score = float(np.clip((trend_points / 4.0) * 100.0, 0.0, 100.0))
    momentum_score = float(np.clip((momentum_points / 4.0) * 100.0, 0.0, 100.0))
    smc_score = float(np.clip((smc_points / 12.0) * 100.0, 0.0, 100.0))
    reversal_score_pct = float(np.clip((reversal_score / 4.0) * 100.0, 0.0, 100.0))
    market_structure_score = float(np.clip(
        (trend_score * 0.35)
        + (momentum_score * 0.25)
        + (smc_score * 0.25)
        + (reversal_score_pct * 0.15),
        0.0,
        100.0,
    ))
    risk_score = float(np.clip(
        100.0
        - (18.0 if not liquidity_ok else 0.0)
        - (20.0 if not trend_ok else 0.0)
        - (15.0 if not smc_confirmed else 0.0)
        - (12.0 if not cycle_ok else 0.0)
        - (10.0 if not macro_gate_ok else 0.0),
        0.0,
        100.0,
    ))

    return {
        "valid": True,
        "symbol": None,
        "decision": decision,
        "score": float(final_score),
        "core_score": float(core_score),
        "market_structure_score": float(market_structure_score),
        "trend_score": float(trend_score),
        "momentum_score": float(momentum_score),
        "smc_score": float(smc_score),
        "reversal_score_pct": float(reversal_score_pct),
        "risk_score": float(risk_score),
        "close": float(last["Close"]),
        "rsi": float(last["RSI14"]),
        "adx": float(last["ADX14"]) if pd.notna(last["ADX14"]) else np.nan,
        "rel_vol": float(last["REL_VOL"]) if pd.notna(last["REL_VOL"]) else np.nan,
        "smart_money_score": float(smart_money_score),
        "cmf20": float(last["CMF20"]) if pd.notna(last["CMF20"]) else np.nan,
        "mfi14": float(last["MFI14"]) if pd.notna(last["MFI14"]) else np.nan,
        "stoch_k": float(last["STOCH_K"]) if pd.notna(last["STOCH_K"]) else np.nan,
        "stoch_d": float(last["STOCH_D"]) if pd.notna(last["STOCH_D"]) else np.nan,
        "cci20": float(last["CCI20"]) if pd.notna(last["CCI20"]) else np.nan,
        "roc12": float(last["ROC12"]) if pd.notna(last["ROC12"]) else np.nan,
        "dominant_period": int(dominant_period),
        "time_to_bottom": int(time_to_next_bottom),
        "time_to_top": int(time_to_next_top),
        "phase_age_bars": phase_age_bars,
        "phase_age_pct": phase_age_pct,
        "cycle_reliability": cycle_reliability,
        "cycle_gate_reason": cycle_info.get("cycle_gate_reason", ""),
        "cycle_info": cycle_info,
        "macro_symbol": macro_symbol,
        "macro_phase": macro_phase,
        "macro_score": macro_score,
        "macro_gate_ok": macro_gate_ok,
        "macro_gate_reason": macro_gate_reason,
        "macro_multiplier": macro_multiplier,
        "macro_cycle_reliability": macro_cycle_reliability,
        "macro_time_to_bottom": macro_time_to_bottom,
        "macro_time_to_top": macro_time_to_top,
        "macro_phase_age_bars": macro_phase_age_bars,
        "macro_phase_age_pct": macro_phase_age_pct,
        "future_fundamental_score": float(future_fundamental_score) if pd.notna(future_fundamental_score) else np.nan,
        "future_fundamental_grade": future_fundamental_grade,
        "future_fundamental_direction": future_fundamental_direction,
        "future_fundamental_confidence": float(future_fundamental_confidence) if pd.notna(future_fundamental_confidence) else np.nan,
        "future_fundamental_phase": future_fundamental_phase,
        "future_fundamental_reason": future_fundamental_reason,
        "phase": phase_info["phase"],
        "phase_confidence": float(phase_info["phase_confidence"]),
        "phase_rank": float(phase_info["phase_rank"]),
        "phase_reason": phase_info["phase_reason"],
        "phase_scores": phase_info["phase_scores"],
        "liquidity_ok": liquidity_ok,
        "trend_ok": trend_ok,
        "fvg_present": bool(d["Bullish_FVG"].tail(5).any()),
        "ob_present": bool(d["Bullish_OB"].tail(5).any()),
        "reversal_score": int(reversal_score),
        "reversal_hits": ", ".join(reversal_hits) if reversal_hits else "-",
        "obv_trend": obv_trend,
        "obv_slope10": obv_slope,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "notes": ",".join(notes) if notes else "SMC_Structure_Clear",
        "df": d,
        "last": last,
    }


def build_entry_plan(
    stock_res: dict,
    entry_buffer_atr: float = 0.25,
    stop_loss_atr: float = 1.8,
    target_1_atr: float = 2.2,
    target_2_atr: float = 3.8,
) -> dict:
    """Build a practical trade plan from the latest analysis output."""
    d = stock_res.get("df")
    last = stock_res.get("last")
    decision = str(stock_res.get("decision", "AVOID"))

    empty_plan = {
        "entry_zone_low": np.nan,
        "entry_zone_high": np.nan,
        "entry_price_plan": np.nan,
        "stop_loss_plan": np.nan,
        "target_1": np.nan,
        "target_2": np.nan,
        "risk_per_share": np.nan,
        "risk_reward_1": np.nan,
        "risk_reward_2": np.nan,
        "upside_to_t1_pct": np.nan,
        "upside_to_t2_pct": np.nan,
        "plan_reason": "No actionable buy signal",
    }

    if decision not in {"BUY", "STRONG BUY"} or d is None or last is None or getattr(d, "empty", True):
        return empty_plan

    try:
        atr_v = _safe_float(last.get("ATR14"), np.nan)
        close = _safe_float(last.get("Close"), np.nan)
        if not np.isfinite(close) or close <= 0:
            return empty_plan

        if not np.isfinite(atr_v) or atr_v <= 0:
            atr_v = max(close * 0.02, 1.0)

        recent_swing_low = float(d["Low"].tail(10).min())
        recent_swing_high = float(d["High"].tail(20).max())

        entry_price = _safe_float(stock_res.get("entry_price"), np.nan)
        stop_price = _safe_float(stock_res.get("stop_price"), np.nan)

        if not np.isfinite(entry_price):
            entry_price = float(close)

        if not np.isfinite(stop_price):
            stop_price = float(min(recent_swing_low, close - atr_v * stop_loss_atr))
            stop_price = max(stop_price, 0.0)

        entry_zone_low = max(0.0, float(min(entry_price, close - atr_v * entry_buffer_atr)))
        entry_zone_high = max(entry_price, float(close + atr_v * max(0.10, entry_buffer_atr * 0.50)))

        risk_per_share = float(max(entry_price - stop_price, 1e-9))
        target_1 = float(max(recent_swing_high, entry_price + atr_v * target_1_atr))
        target_2 = float(max(target_1 + atr_v * 0.80, entry_price + atr_v * target_2_atr))

        rr1 = float((target_1 - entry_price) / risk_per_share)
        rr2 = float((target_2 - entry_price) / risk_per_share)
        upside_t1 = float((target_1 / entry_price - 1.0) * 100.0)
        upside_t2 = float((target_2 / entry_price - 1.0) * 100.0)

        return {
            "entry_zone_low": entry_zone_low,
            "entry_zone_high": entry_zone_high,
            "entry_price_plan": entry_price,
            "stop_loss_plan": stop_price,
            "target_1": target_1,
            "target_2": target_2,
            "risk_per_share": risk_per_share,
            "risk_reward_1": rr1,
            "risk_reward_2": rr2,
            "upside_to_t1_pct": upside_t1,
            "upside_to_t2_pct": upside_t2,
            "plan_reason": "Auto plan built from trend, ATR, structure, and recent swing levels",
        }
    except Exception as e:
        empty_plan["plan_reason"] = f"Plan error: {e}"
        return empty_plan


# =========================================================
# Universe loading
# =========================================================
if universe_mode == "Paste tickers":
    universe = parse_universe_text(paste_text)
elif universe_mode == "Upload CSV":
    universe = load_universe_from_csv(uploaded_file)
else:
    local_file = Path("midcap_universe.csv")
    universe = load_universe_from_csv(local_file) if local_file.exists() else []

if "global_scan_results" not in st.session_state:
    st.session_state.global_scan_results = []
if "global_watch_df" not in st.session_state:
    st.session_state.global_watch_df = pd.DataFrame()
if "global_valid_results" not in st.session_state:
    st.session_state.global_valid_results = []

flow_val = map_flow_to_score("Netral")
GLOBAL_BENCHMARK_SYMBOL = "^JKSE"
GLOBAL_BENCHMARK_DF = load_ticker_data(GLOBAL_BENCHMARK_SYMBOL, months)
GLOBAL_MACRO_CONTEXT = build_macro_liquidity_gate(GLOBAL_BENCHMARK_DF, GLOBAL_BENCHMARK_SYMBOL)


def process_symbol(symbol: str):
    try:
        d = load_ticker_data(symbol, months)
        if d.empty or len(d) < min_history_bars:
            return {"valid": False, "symbol": symbol, "reason": "Data historis tidak mencukupi"}

        fundamental = compute_fundamental_grade(symbol)
        future_context = compute_future_fundamental_grade(symbol, d, GLOBAL_MACRO_CONTEXT)
        res = score_stock_smc(
            d,
            flow_used=False,
            flow_val=50,
            min_avg_volume=min_avg_volume,
            min_price=min_price,
            max_price=max_price,
            mode=GLOBAL_MODE,
            min_history_bars=min_history_bars,
            macro_context=GLOBAL_MACRO_CONTEXT,
            future_fundamental_context=future_context,
        )
        res["entry_plan"] = build_entry_plan(res)
        res.update(res["entry_plan"])
        ifs_context = compute_institutional_forward_score(
            symbol=symbol,
            price_df=d,
            bench_df=GLOBAL_MACRO_CONTEXT.get("benchmark_df"),
            current_fundamental=fundamental,
            future_context=future_context,
            technical_context=res,
        )
        res["symbol"] = symbol
        res["fundamental_score"] = fundamental.get("fundamental_score", np.nan)
        res["fundamental_grade"] = fundamental.get("fundamental_grade", "n/a")
        res["ifs_score"] = ifs_context.get("ifs_score", np.nan)
        res["ifs_grade"] = ifs_context.get("ifs_grade", "n/a")
        res["ifs_breakdown"] = ifs_context.get("ifs_breakdown", {})
        res["ifs_detail"] = ifs_context.get("ifs_detail", {})
        return res
    except Exception as e:
        return {"valid": False, "symbol": symbol, "reason": str(e)}


# =========================================================

def run_deep_dive_analysis(
    ticker_input: str,
    strategy_mode: str,
    bandarmology_mode: str,
    benchmark_symbol_local: str,
    show_benchmark_local: bool,
    entry_buffer_atr_local: float,
    stop_loss_atr_local: float,
    take_profit_1_atr_local: float,
    take_profit_2_atr_local: float,
) -> dict:
    """Run a single-ticker deep dive and return a reusable analysis bundle."""
    deep_ticker = normalize_ticker(ticker_input)
    flow_val_local = map_flow_to_score(bandarmology_mode)

    stock_df = load_ticker_data(deep_ticker, months)
    bench_df = load_ticker_data(benchmark_symbol_local, months) if benchmark_symbol_local else pd.DataFrame()

    macro_context = None
    if show_benchmark_local and not bench_df.empty and len(bench_df) >= min_history_bars:
        macro_context = build_macro_liquidity_gate(bench_df.copy(), benchmark_symbol_local)

    if stock_df.empty or len(stock_df) < min_history_bars:
        return {
            "symbol": deep_ticker,
            "stock_df": stock_df,
            "bench_df": bench_df,
            "macro_context": macro_context,
            "stock_res": None,
            "fundamental": None,
            "future_context": None,
            "ifs_context": None,
            "entry_plan": None,
            "error": "Data ticker tidak cukup atau gagal diunduh.",
        }

    future_fundamental_context = compute_future_fundamental_grade(deep_ticker, stock_df, macro_context)
    stock_res = score_stock_smc(
        stock_df,
        flow_used=True,
        flow_val=flow_val_local,
        min_avg_volume=min_avg_volume,
        min_price=min_price,
        max_price=max_price,
        mode=strategy_mode,
        min_history_bars=min_history_bars,
        macro_context=macro_context,
        future_fundamental_context=future_fundamental_context,
    )
    fundamental = compute_fundamental_grade(deep_ticker)
    stock_res["peg_ratio"] = fundamental.get("peg_ratio", np.nan)
    stock_res["trailing_pe"] = fundamental.get("trailing_pe", np.nan)
    stock_res["forward_pe"] = fundamental.get("forward_pe", np.nan)
    stock_res["revenue_growth"] = fundamental.get("revenue_growth", np.nan)
    stock_res["earnings_growth"] = fundamental.get("earnings_growth", np.nan)
    stock_res["profit_margins"] = fundamental.get("profit_margins", np.nan)
    stock_res["future_fundamental_score"] = future_fundamental_context.get("future_fundamental_score", np.nan)
    stock_res["future_fundamental_grade"] = future_fundamental_context.get("future_fundamental_grade", "n/a")
    stock_res["future_fundamental_direction"] = future_fundamental_context.get("future_fundamental_direction", "n/a")
    stock_res["future_fundamental_confidence"] = future_fundamental_context.get("future_fundamental_confidence", np.nan)
    stock_res["future_fundamental_phase"] = future_fundamental_context.get("future_phase", "Unknown")
    stock_res["future_fundamental_reason"] = future_fundamental_context.get("future_moat_reason", "n/a")

    entry_plan = build_entry_plan(
        stock_res,
        entry_buffer_atr=entry_buffer_atr_local,
        stop_loss_atr=stop_loss_atr_local,
        target_1_atr=take_profit_1_atr_local,
        target_2_atr=take_profit_2_atr_local,
    )
    stock_res["entry_plan"] = entry_plan
    stock_res.update(entry_plan)

    ifs_context = compute_institutional_forward_score(
        symbol=deep_ticker,
        price_df=stock_df,
        bench_df=bench_df,
        current_fundamental=fundamental,
        future_context=future_fundamental_context,
        technical_context=stock_res,
    )
    stock_res["ifs_score"] = ifs_context.get("ifs_score", np.nan)
    stock_res["ifs_grade"] = ifs_context.get("ifs_grade", "n/a")
    stock_res["ifs_breakdown"] = ifs_context.get("ifs_breakdown", {})
    stock_res["ifs_detail"] = ifs_context.get("ifs_detail", {})

    return {
        "symbol": deep_ticker,
        "stock_df": stock_df,
        "bench_df": bench_df,
        "macro_context": macro_context,
        "stock_res": stock_res,
        "fundamental": fundamental,
        "future_context": future_fundamental_context,
        "ifs_context": ifs_context,
        "entry_plan": entry_plan,
    }




# =========================================================
# Tabs
# =========================================================
tab1, tab2 = st.tabs(["📈 Market Structure", "🏦 Institutional Forward Score"])

with tab1:
    st.subheader("Market Structure Top 20")
    st.caption("Fokus pada trend, momentum, cycle, risk, dan setup teknikal yang paling kuat.")

    if run_global_scan:
        if not universe:
            st.error("Universe kosong. Isi tickers di sidebar terlebih dahulu.")
        else:
            st.write(f"⚙️ Memproses analisis struktural pada **{len(universe)}** emiten...")
            progress = st.progress(0)
            status = st.empty()
            results = []

            with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(process_symbol, sym): sym for sym in universe}
                done = 0
                total = len(futures)
                for fut in cf.as_completed(futures):
                    done += 1
                    progress.progress(done / total)
                    status.caption(f"Selesai mengurai: {done}/{total} -> {futures[fut]}")
                    results.append(fut.result())

            progress.empty()
            status.empty()

            st.session_state.global_scan_results = results
            valid_results = [r for r in results if r.get("valid")]
            st.session_state.global_valid_results = valid_results

            if not valid_results:
                st.error("Tidak ada emiten yang lolos filter data dasar.")
                reasons = pd.DataFrame(
                    [{"Ticker": r.get("symbol"), "Reason": r.get("reason", "-")} for r in results if not r.get("valid")]
                )
                if not reasons.empty:
                    st.dataframe(reasons, use_container_width=True, hide_index=True)
            else:
                watch_rows = []
                for r in valid_results:
                    watch_rows.append(
                        {
                            "Ticker": r["symbol"],
                            "Decision": r["decision"],
                            "Score": round(r["score"], 2),
                            "MarketStruct": round(r.get("market_structure_score", np.nan), 2) if pd.notna(r.get("market_structure_score", np.nan)) else np.nan,
                            "Trend": round(r.get("trend_score", np.nan), 1) if pd.notna(r.get("trend_score", np.nan)) else np.nan,
                            "Momentum": round(r.get("momentum_score", np.nan), 1) if pd.notna(r.get("momentum_score", np.nan)) else np.nan,
                            "Cycle": r.get("dominant_period", np.nan),
                            "CycleRel": round(r.get("cycle_reliability", np.nan), 1) if pd.notna(r.get("cycle_reliability", np.nan)) else np.nan,
                            "Risk": round(r.get("risk_score", np.nan), 1) if pd.notna(r.get("risk_score", np.nan)) else np.nan,
                            "SmartMoney": round(r.get("smart_money_score", np.nan), 2) if pd.notna(r.get("smart_money_score", np.nan)) else np.nan,
                            "Reversal": r["reversal_hits"],
                            "FVG": "🔥 YES" if r["fvg_present"] else "NO",
                            "OrderBlock": "🎯 YES" if r["ob_present"] else "NO",
                            "TrendState": "BULLISH" if r["trend_ok"] else "BEARISH",
                            "Phase": r.get("phase", "-"),
                            "PhaseConf": round(r.get("phase_confidence", np.nan), 0) if pd.notna(r.get("phase_confidence", np.nan)) else np.nan,
                            "MacroPhase": r.get("macro_phase", "-"),
                            "MacroScore": round(r.get("macro_score", np.nan), 1) if pd.notna(r.get("macro_score", np.nan)) else np.nan,
                            "MacroGate": "ON" if r.get("macro_gate_ok", True) else "OFF",
                            "IFS": round(r.get("ifs_score", np.nan), 2) if pd.notna(r.get("ifs_score", np.nan)) else np.nan,
                            "IFSGrade": r.get("ifs_grade", "n/a"),
                            "Entry": round(r["entry_price"], 2) if pd.notna(r["entry_price"]) else np.nan,
                            "Stop": round(r["stop_price"], 2) if pd.notna(r["stop_price"]) else np.nan,
                            "TP1": round(r.get("target_1", np.nan), 2) if pd.notna(r.get("target_1", np.nan)) else np.nan,
                            "TP2": round(r.get("target_2", np.nan), 2) if pd.notna(r.get("target_2", np.nan)) else np.nan,
                            "RR1": round(r.get("risk_reward_1", np.nan), 2) if pd.notna(r.get("risk_reward_1", np.nan)) else np.nan,
                            "RR2": round(r.get("risk_reward_2", np.nan), 2) if pd.notna(r.get("risk_reward_2", np.nan)) else np.nan,
                            "Notes": r["notes"],
                        }
                    )

                watch_df = (
                    pd.DataFrame(watch_rows)
                    .sort_values(
                        ["MarketStruct", "IFS", "Score", "SmartMoney", "CycleRel"],
                        ascending=[False, False, False, False, False],
                        na_position="last",
                    )
                    .reset_index(drop=True)
                )
                st.session_state.global_watch_df = watch_df

                top20 = watch_df.head(20).copy()

                st.subheader("🔥 Top 3 High-Conviction Setups")
                top3 = top20[top20["Decision"].isin(["BUY", "STRONG BUY"])].head(3)
                if not top3.empty:
                    cols = st.columns(len(top3))
                    for idx, row in enumerate(top3.itertuples()):
                        with cols[idx]:
                            st.metric(
                                label=f"🌟 {row.Ticker} ({row.Decision})",
                                value=f"Rp {row.Entry:,.0f}" if pd.notna(row.Entry) else f"Rp {row.Stop:,.0f}",
                                delta=f"IFS: {row.IFS}",
                            )
                            st.markdown(
                                f"**Market Struct:** `{row.MarketStruct}`  \n"
                                f"**Trend/Momentum:** `{row.Trend}` / `{row.Momentum}`  \n"
                                f"**Cycle:** `{row.Cycle}` bars | Rel `{row.CycleRel}`  \n"
                                f"**Risk:** `{row.Risk}`  \n"
                                f"**TP1/TP2:** `{row.TP1}` / `{row.TP2}`  \n"
                                f"**RR1/RR2:** `{row.RR1}` / `{row.RR2}`  \n"
                                f"**Smart Money:** `{row.SmartMoney}`  \n"
                                f"**Phase:** `{row.Phase}`"
                            )
                else:
                    st.info("Belum ada kandidat BUY/STRONG BUY pada universe saat ini.")

                st.markdown("---")
                st.subheader("🏆 Market Structure Ranking (Top 20)")
                st.dataframe(top20, use_container_width=True, hide_index=True)
    else:
        if not st.session_state.global_watch_df.empty:
            st.subheader("🏆 Market Structure Ranking (Top 20)")
            st.dataframe(st.session_state.global_watch_df.head(20), use_container_width=True, hide_index=True)
            st.info("Klik **Run global scan** di sidebar untuk memperbarui ranking.")
        else:
            st.info("Klik **Run global scan** di sidebar untuk mulai scan universe.")

with tab2:
    st.subheader("🏦 Institutional Forward Score")
    st.caption("Dibagi menjadi overview, factor breakdown, smart money, forward fundamental, entry plan, dan detail saham.")

    c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
    with c1:
        ticker_input = st.text_input("Ticker saham", value="BMRI", key="deep_ticker_input")
    with c2:
        strategy_mode = st.selectbox(
            "Strategy mode",
            ["Conservative", "Balanced", "Aggressive"],
            index=1,
            key="deep_strategy_mode",
        )
    with c3:
        bandarmology_mode = st.selectbox(
            "Bandarmology",
            ["Big Akumulasi", "Small Akumulasi", "Netral", "Small Distribusi", "Big Distribusi"],
            index=2,
            key="deep_bandarmology_mode",
        )

    with st.expander("⚙️ Deep Dive Settings", expanded=True):
        d1, d2, d3, d4 = st.columns([1, 1, 1, 1])
        with d1:
            benchmark_symbol_local = st.text_input("Benchmark IHSG symbol", value="^JKSE", key="deep_benchmark_symbol")
        with d2:
            show_benchmark_local = st.checkbox("Tampilkan benchmark vs saham", value=True, key="deep_show_benchmark")
        with d3:
            entry_buffer_atr_local = st.slider("Entry buffer (x ATR)", 0.10, 1.00, 0.25, 0.05, key="deep_entry_buffer_atr")
        with d4:
            stop_loss_atr_local = st.slider("Stop Loss (x ATR)", 1.0, 5.0, 1.8, 0.1, key="deep_stop_loss_atr")

        d5, d6 = st.columns([1, 1])
        with d5:
            take_profit_1_atr_local = st.slider("Take Profit 1 (x ATR)", 1.0, 6.0, 2.2, 0.1, key="deep_take_profit_1_atr")
        with d6:
            take_profit_2_atr_local = st.slider("Take Profit 2 (x ATR)", 2.0, 8.0, 3.8, 0.1, key="deep_take_profit_2_atr")

        analyze_btn = st.button("Analyze ticker", type="primary", key="deep_analyze_btn")

    analysis_bundle = {}
    if analyze_btn:
        deep_ticker = normalize_ticker(ticker_input)
        analysis_bundle = run_deep_dive_analysis(
            ticker_input=ticker_input,
            strategy_mode=strategy_mode,
            bandarmology_mode=bandarmology_mode,
            benchmark_symbol_local=benchmark_symbol_local,
            show_benchmark_local=show_benchmark_local,
            entry_buffer_atr_local=entry_buffer_atr_local,
            stop_loss_atr_local=stop_loss_atr_local,
            take_profit_1_atr_local=take_profit_1_atr_local,
            take_profit_2_atr_local=take_profit_2_atr_local,
        )
        st.session_state.ifs_analysis = analysis_bundle
    else:
        analysis_bundle = st.session_state.get("ifs_analysis", {})

    stock_res = analysis_bundle.get("stock_res")
    ifs_context = analysis_bundle.get("ifs_context")
    fundamental = analysis_bundle.get("fundamental")
    future_context = analysis_bundle.get("future_context")
    deep_ticker = analysis_bundle.get("symbol", normalize_ticker(ticker_input))
    stock_df = analysis_bundle.get("stock_df", pd.DataFrame())
    bench_df = analysis_bundle.get("bench_df", pd.DataFrame())
    macro_context = analysis_bundle.get("macro_context")
    entry_plan = analysis_bundle.get("entry_plan", {})

    sub_overview, sub_factor, sub_smart, sub_forward, sub_entry, sub_detail = st.tabs(
        ["Overview", "Factor Breakdown", "Smart Money", "Forward Fundamental", "Entry Plan", "Detail Saham"]
    )

    with sub_entry:
        st.subheader("Entry Plan")
        if ifs_context is not None and stock_res is not None:
            plan = stock_res.get("entry_plan", {})
            cols = st.columns(4)
            cols[0].metric("Signal", stock_res.get("decision", "n/a"), f'Confidence {ifs_context.get("ifs_detail", {}).get("future_confidence", np.nan):.0f}%')
            cols[1].metric("Entry", f'Rp {plan.get("entry_price_plan", np.nan):,.0f}' if pd.notna(plan.get("entry_price_plan", np.nan)) else "n/a")
            cols[2].metric("Stop Loss", f'Rp {plan.get("stop_loss_plan", np.nan):,.0f}' if pd.notna(plan.get("stop_loss_plan", np.nan)) else "n/a")
            cols[3].metric("RR2", f'{plan.get("risk_reward_2", np.nan):.2f}' if pd.notna(plan.get("risk_reward_2", np.nan)) else "n/a")

            entry_table = pd.DataFrame(
                [
                    {"Metric": "Decision", "Value": stock_res.get("decision", "n/a")},
                    {"Metric": "Entry Zone Low", "Value": f'Rp {plan.get("entry_zone_low", np.nan):,.0f}' if pd.notna(plan.get("entry_zone_low", np.nan)) else "n/a"},
                    {"Metric": "Entry Zone High", "Value": f'Rp {plan.get("entry_zone_high", np.nan):,.0f}' if pd.notna(plan.get("entry_zone_high", np.nan)) else "n/a"},
                    {"Metric": "Stop Loss", "Value": f'Rp {plan.get("stop_loss_plan", np.nan):,.0f}' if pd.notna(plan.get("stop_loss_plan", np.nan)) else "n/a"},
                    {"Metric": "Target 1", "Value": f'Rp {plan.get("target_1", np.nan):,.0f}' if pd.notna(plan.get("target_1", np.nan)) else "n/a"},
                    {"Metric": "Target 2", "Value": f'Rp {plan.get("target_2", np.nan):,.0f}' if pd.notna(plan.get("target_2", np.nan)) else "n/a"},
                    {"Metric": "Risk / Share", "Value": f'Rp {plan.get("risk_per_share", np.nan):,.0f}' if pd.notna(plan.get("risk_per_share", np.nan)) else "n/a"},
                    {"Metric": "RR1", "Value": f'{plan.get("risk_reward_1", np.nan):.2f}' if pd.notna(plan.get("risk_reward_1", np.nan)) else "n/a"},
                    {"Metric": "RR2", "Value": f'{plan.get("risk_reward_2", np.nan):.2f}' if pd.notna(plan.get("risk_reward_2", np.nan)) else "n/a"},
                    {"Metric": "Upside TP1", "Value": f'{plan.get("upside_to_t1_pct", np.nan):.2f}%' if pd.notna(plan.get("upside_to_t1_pct", np.nan)) else "n/a"},
                    {"Metric": "Upside TP2", "Value": f'{plan.get("upside_to_t2_pct", np.nan):.2f}%' if pd.notna(plan.get("upside_to_t2_pct", np.nan)) else "n/a"},
                    {"Metric": "Plan Reason", "Value": plan.get("plan_reason", "n/a")},
                ]
            )
            st.dataframe(entry_table, use_container_width=True, hide_index=True)
        else:
            st.info("Klik Analyze ticker untuk melihat entry plan otomatis.")

    with sub_detail:
        if analyze_btn:
            deep_ticker = normalize_ticker(ticker_input)
            flow_val_local = map_flow_to_score(bandarmology_mode)

            stock_df = load_ticker_data(deep_ticker, months)
            bench_df = load_ticker_data(benchmark_symbol_local, months) if benchmark_symbol_local else pd.DataFrame()

            macro_context = None
            if show_benchmark_local and not bench_df.empty and len(bench_df) >= min_history_bars:
                macro_context = build_macro_liquidity_gate(bench_df.copy(), benchmark_symbol_local)

            if stock_df.empty or len(stock_df) < min_history_bars:
                st.error("Data ticker tidak cukup atau gagal diunduh.")
            else:
                future_fundamental_context = compute_future_fundamental_grade(deep_ticker, stock_df, macro_context)

                stock_res = score_stock_smc(
                    stock_df,
                    flow_used=True,
                    flow_val=flow_val_local,
                    min_avg_volume=min_avg_volume,
                    min_price=min_price,
                    max_price=max_price,
                    mode=strategy_mode,
                    min_history_bars=min_history_bars,
                    macro_context=macro_context,
                    future_fundamental_context=future_fundamental_context,
                )

                stock = stock_res["df"].copy()
                stock_last = stock_res["last"]
                fundamental = compute_fundamental_grade(deep_ticker)
                stock_res["peg_ratio"] = fundamental.get("peg_ratio", np.nan)
                stock_res["trailing_pe"] = fundamental.get("trailing_pe", np.nan)
                stock_res["forward_pe"] = fundamental.get("forward_pe", np.nan)
                stock_res["revenue_growth"] = fundamental.get("revenue_growth", np.nan)
                stock_res["earnings_growth"] = fundamental.get("earnings_growth", np.nan)
                stock_res["profit_margins"] = fundamental.get("profit_margins", np.nan)
                stock_res["future_fundamental_score"] = future_fundamental_context.get("future_fundamental_score", np.nan)
                stock_res["future_fundamental_grade"] = future_fundamental_context.get("future_fundamental_grade", "n/a")
                stock_res["future_fundamental_direction"] = future_fundamental_context.get("future_fundamental_direction", "n/a")
                stock_res["future_fundamental_confidence"] = future_fundamental_context.get("future_fundamental_confidence", np.nan)
                stock_res["future_fundamental_phase"] = future_fundamental_context.get("future_phase", "Unknown")
                stock_res["future_fundamental_reason"] = future_fundamental_context.get("future_moat_reason", "n/a")
                ifs_context = compute_institutional_forward_score(
                    symbol=deep_ticker,
                    price_df=stock_df,
                    bench_df=bench_df,
                    current_fundamental=fundamental,
                    future_context=future_fundamental_context,
                    technical_context=stock_res,
                )
                entry_plan = build_entry_plan(
                    stock_res,
                    entry_buffer_atr=entry_buffer_atr_local,
                    stop_loss_atr=stop_loss_atr_local,
                    target_1_atr=take_profit_1_atr_local,
                    target_2_atr=take_profit_2_atr_local,
                )
                stock_res["entry_plan"] = entry_plan
                stock_res.update(entry_plan)
                stock_res["ifs_score"] = ifs_context.get("ifs_score", np.nan)
                stock_res["ifs_grade"] = ifs_context.get("ifs_grade", "n/a")
                stock_res["ifs_breakdown"] = ifs_context.get("ifs_breakdown", {})
                stock_res["ifs_detail"] = ifs_context.get("ifs_detail", {})
                st.session_state.ifs_analysis = {
                    "symbol": deep_ticker,
                    "stock_df": stock_df,
                    "bench_df": bench_df,
                    "macro_context": macro_context,
                    "stock_res": stock_res,
                    "fundamental": fundamental,
                    "future_context": future_fundamental_context,
                    "ifs_context": ifs_context,
                    "entry_plan": entry_plan,
                    "strategy_mode": strategy_mode,
                    "bandarmology_mode": bandarmology_mode,
                    "ticker_input": ticker_input,
                }
                bench = pd.DataFrame()
                bench_cycle = None
                if macro_context is not None:
                    bench = bench_df.copy()
                    bench_cycle = macro_context.get("cycle_tuple")

                stock_status = "Near Bottom" if stock_res["time_to_bottom"] <= 4 else "Mid-Cycle Moving"
                bench_status = "n/a"
                if macro_context is not None:
                    bench_status = "Near Bottom" if macro_context.get("macro_time_to_bottom", 999) <= 4 else "Mid-Cycle Moving"
                stock_top = stock_res.get("time_to_top", np.nan)
                stock_phase_age = stock_res.get("phase_age_bars", np.nan)
                stock_phase_age_pct = stock_res.get("phase_age_pct", np.nan)
                macro_context = macro_context or build_macro_liquidity_gate(pd.DataFrame(), benchmark_symbol_local)
                bench_top = macro_context.get("macro_time_to_top", np.nan) if macro_context is not None else np.nan
                bench_phase_age = macro_context.get("macro_phase_age_bars", np.nan) if macro_context is not None else np.nan
                bench_phase_age_pct = macro_context.get("macro_phase_age_pct", np.nan) if macro_context is not None else np.nan

                st.markdown(
                    """
                    <div style="margin-top: 0.25rem;">
                        <h2 style="margin-bottom:0.25rem;">⏳ Trader Time Analysis Model</h2>
                        <div style="font-size:1.05rem; opacity:0.9;">
                            Mengukur frekuensi dominan dan estimasi waktu pembalikan tren berlandaskan struktur matematika siklus bursa.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                stock_period = stock_res["dominant_period"]
                stock_ttb = stock_res["time_to_bottom"]
                stock_cycle_info = stock_res.get("cycle_info", {})
                bench_period = bench_cycle[0] if bench_cycle is not None else None
                bench_ttb = bench_cycle[1] if bench_cycle is not None else None
                bench_cycle_info = bench_cycle[3] if bench_cycle is not None and len(bench_cycle) > 3 else {}

                stock_html = f"""
                <div style="background:linear-gradient(180deg, rgba(235,244,255,1) 0%, rgba(225,235,250,1) 100%); padding:22px; border-radius:18px; border:1px solid rgba(0,0,0,0.05); box-shadow:0 8px 24px rgba(0,0,0,0.04);">
                    <div style="font-size:1.15rem; font-weight:700; color:#173b6d; margin-bottom:18px;">Siklus Saham ({deep_ticker})</div>
                    <div style="font-size:1.02rem; color:#173b6d; line-height:2;">
                        <div>• <b>Periode Siklus Dominan:</b> {stock_period} Hari Bursa</div>
                        <div>• <b>Estimasi Sisa Waktu Menuju Bottom berikutnya:</b> {stock_ttb} Bar</div>
                        <div>• <b>Estimasi Menuju Top Berikutnya:</b> {stock_top} Bar</div>
                        <div>• <b>Phase Age:</b> {stock_phase_age} Bar ({stock_phase_age_pct:.0f}%)</div>
                        <div>• <b>Status Posisi Siklus:</b> {stock_status}</div>
                        <div>• <b>8-Phase Cycle:</b> {stock_res["phase"]} ({stock_res["phase_confidence"]:.0f}%)</div>
                        <div>• <b>FFT / Hilbert / Autocorr:</b> {stock_cycle_info.get("fft_period", "-")} / {stock_cycle_info.get("hilbert_period", "-")} / {stock_cycle_info.get("autocorr_period", "-")}</div>
                        <div>• <b>Weighted Composite:</b> {stock_cycle_info.get("weighted_period", stock_period)} bars</div>
                        <div>• <b>Cycle Reliability:</b> {stock_cycle_info.get("cycle_reliability", np.nan):.0f}%</div>
                        <div>• <b>Cycle Gate Reason:</b> {stock_cycle_info.get("cycle_gate_reason", "OK")}</div>
                        <div>• <b>Detrend Method:</b> {stock_cycle_info.get('detrend_method', 'HighPass+TailHilbert')}</div>
                        <div>• <b>Macro Gate:</b> {'ON' if stock_res.get('macro_gate_ok', True) else 'OFF'} ({stock_res.get('macro_phase', 'Unknown')})</div>
                        <div>• <b>Macro Gate Reason:</b> {stock_res.get('macro_gate_reason', 'OK')}</div>
                        <div>• <b>Trend Lag:</b> {stock_cycle_info.get('trend_lag_bars', '-')} Bar</div>
                    </div>
                </div>
                """
                bench_html = f"""
                <div style="background:linear-gradient(180deg, rgba(255,248,230,1) 0%, rgba(248,238,210,1) 100%); padding:22px; border-radius:18px; border:1px solid rgba(0,0,0,0.05); box-shadow:0 8px 24px rgba(0,0,0,0.04);">
                    <div style="font-size:1.15rem; font-weight:700; color:#8a4b00; margin-bottom:18px;">Siklus Makro Komposit (IHSG)</div>
                    <div style="font-size:1.02rem; color:#8a4b00; line-height:2;">
                        <div>• <b>Periode Siklus Dominan:</b> {bench_period if bench_period is not None else '-'} Hari Bursa</div>
                        <div>• <b>Estimasi Sisa Waktu Menuju Bottom berikutnya:</b> {bench_ttb if bench_ttb is not None else '-'} Bar</div>
                        <div>• <b>Status Posisi Siklus Makro:</b> {bench_status}</div>
                        <div>• <b>FFT / Hilbert / Autocorr:</b> {bench_cycle_info.get("fft_period", "-")} / {bench_cycle_info.get("hilbert_period", "-")} / {bench_cycle_info.get("autocorr_period", "-")}</div>
                        <div>• <b>Weighted Composite:</b> {bench_cycle_info.get("weighted_period", bench_period if bench_period is not None else '-') } bars</div>
                        <div>• <b>Estimasi Menuju Top Berikutnya:</b> {bench_top} Bar</div>
                        <div>• <b>Phase Age:</b> {bench_phase_age} Bar ({bench_phase_age_pct:.0f}%)</div>
                        <div>• <b>Macro Score:</b> {macro_context.get('macro_score', np.nan):.0f}%</div>
                        <div>• <b>Macro Gate:</b> {'ON' if macro_context.get('macro_gate_ok', True) else 'OFF'}</div>
                        <div>• <b>Macro Gate Reason:</b> {macro_context.get('macro_gate_reason', 'OK')}</div>
                        <div>• <b>Detrend Method:</b> {bench_cycle_info.get('detrend_method', 'ZLEMA')}</div>
                        <div>• <b>Trend Lag:</b> {bench_cycle_info.get('trend_lag_bars', '-')} Bar</div>
                    </div>
                </div>
                """
                st.markdown(
                    f"""
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:18px; margin: 18px 0 8px 0;">
                        {stock_html}
                        {bench_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                ctop1, ctop2, ctop3, ctop4 = st.columns(4)
                ctop1.metric("Decision", stock_res["decision"])
                ctop2.metric("Score", f"{stock_res['score']:.2f}")
                ctop3.metric("Close", f"Rp {stock_res['close']:,.0f}")
                ctop4.metric("Phase", stock_res["phase"])

                ctop5, ctop6, ctop7, ctop8 = st.columns(4)
                ctop5.metric("Smart Money", f"{stock_res['smart_money_score']:.0f}")
                ctop6.metric("Fundamental", f"{fundamental.get('fundamental_score', np.nan):.0f}" if pd.notna(fundamental.get('fundamental_score', np.nan)) else "n/a")
                ctop7.metric("PEG", f"{fundamental.get('peg_ratio', np.nan):.2f}" if pd.notna(fundamental.get('peg_ratio', np.nan)) else "n/a")
                ctop8.metric("Grade", fundamental.get("fundamental_grade", "n/a"))

                cmid1, cmid2, cmid3, cmid4 = st.columns(4)
                cmid1.metric("Relative Volume", f"{stock_res['rel_vol']:.2f}x" if pd.notna(stock_res["rel_vol"]) else "n/a")
                cmid2.metric("RSI14", f"{stock_res['rsi']:.2f}")
                cmid3.metric("ADX14", f"{stock_res['adx']:.2f}" if pd.notna(stock_res["adx"]) else "n/a")
                cmid4.metric("Phase Confidence", f"{stock_res['phase_confidence']:.0f}%")

                left, right = st.columns([1, 1])
                with left:
                    st.subheader("Time Analysis - Stock")
                    st.write(f"**Dominant cycle:** `{stock_res['dominant_period']} bars`")
                    st.write(f"**FFT / Hilbert / Autocorr:** `{stock_cycle_info.get('fft_period', '-')}` / `{stock_cycle_info.get('hilbert_period', '-')}` / `{stock_cycle_info.get('autocorr_period', '-')}`")
                    st.write(f"**Weighted composite:** `{stock_cycle_info.get('weighted_period', stock_res['dominant_period'])} bars`")
                    st.write(f"**Detrend method:** `{stock_cycle_info.get('detrend_method', 'ZLEMA')}` | lag `{stock_cycle_info.get('trend_lag_bars', '-')}` bars")
                    st.write(f"**Time to next bottom:** `{stock_res['time_to_bottom']} bars`")
                    st.write(f"**Time to next top:** `{stock_res.get('time_to_top', np.nan)} bars`")
                    st.write(f"**Phase age:** `{stock_res.get('phase_age_bars', np.nan)} bars` ({stock_res.get('phase_age_pct', np.nan):.0f}%)")
                    st.write(f"**Cycle status:** `{stock_status}`")
                    st.write(f"**8-Phase:** `{stock_res['phase']}`")
                    st.write(f"**Phase confidence:** `{stock_res['phase_confidence']:.0f}%`")
                    st.write(f"**Phase reason:** {stock_res['phase_reason']}")
                    st.write(f"**Reversal signals:** `{stock_res['reversal_hits']}`")
                    st.write(f"**OBV trend:** `{stock_res['obv_trend']}`")
                    st.write(f"**CMF20 / MFI14:** `{stock_res['cmf20']:.2f}` / `{stock_res['mfi14']:.2f}`")
                    st.write(f"**Stoch K/D:** `{stock_res['stoch_k']:.2f}` / `{stock_res['stoch_d']:.2f}`")
                    st.write(f"**PEG:** `{stock_res.get('peg_ratio', np.nan):.2f}`" if pd.notna(stock_res.get("peg_ratio", np.nan)) else "**PEG:** n/a")
                    st.write(f"**SMC:** FVG `{stock_res['fvg_present']}` | OB `{stock_res['ob_present']}`")
                    st.write(f"**Bandarmology input:** `{bandarmology_mode}`")

                with right:
                    st.subheader("Recommendation")
                    if stock_res["decision"] in {"BUY", "STRONG BUY"}:
                        st.success("Saham layak dibeli menurut filter saat ini.")
                        st.write(f"**Recommended entry:** `Rp {stock_res['entry_price']:,.0f}`")
                        st.write(f"**Recommended stoploss:** `Rp {stock_res['stop_price']:,.0f}`")
                        rr_risk = stock_res["entry_price"] - stock_res["stop_price"]
                        st.write(f"**Risk per share:** `Rp {rr_risk:,.0f}`")
                        tp_price = stock_res["entry_price"] + take_profit_atr_local * float(stock_res["last"]["ATR14"])
                        st.write(f"**Take profit target:** `Rp {tp_price:,.0f}`")
                    else:
                        st.warning("Belum layak beli. Tunggu reversal / struktur membaik.")
                        st.write("Entry/stoploss tidak ditampilkan karena belum memenuhi kriteria beli.")

                st.markdown("---")
                fig = make_subplots(
                    rows=4,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.04,
                    row_heights=[0.45, 0.15, 0.20, 0.20],
                    subplot_titles=(
                        f"{deep_ticker} Price Action",
                        "Reversal / SMC / OBV Signals",
                        "Relative Strength vs Benchmark",
                        "Volume",
                    ),
                )

                fig.add_trace(
                    go.Candlestick(
                        x=stock.index,
                        open=stock["Open"],
                        high=stock["High"],
                        low=stock["Low"],
                        close=stock["Close"],
                        name="Price",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(go.Scatter(x=stock.index, y=stock["EMA20"], name="EMA20", mode="lines"), row=1, col=1)
                fig.add_trace(go.Scatter(x=stock.index, y=stock["EMA50"], name="EMA50", mode="lines"), row=1, col=1)
                fig.add_trace(go.Scatter(x=stock.index, y=stock["EMA200"], name="EMA200", mode="lines"), row=1, col=1)

                fvg_df = stock[stock["Bullish_FVG"]].tail(5)
                for idx, _ in fvg_df.iterrows():
                    loc = stock.index.get_loc(idx)
                    if loc >= 2:
                        fig.add_shape(
                            type="rect",
                            x0=idx,
                            x1=stock.index[-1],
                            y0=float(stock["High"].iloc[loc - 2]),
                            y1=float(stock["Low"].iloc[loc]),
                            fillcolor="rgba(0, 255, 0, 0.08)",
                            line=dict(width=0),
                            row=1,
                            col=1,
                        )

                ob_df = stock[stock["Bullish_OB"]].tail(5)
                for idx, _ in ob_df.iterrows():
                    loc = stock.index.get_loc(idx)
                    if loc >= 1:
                        fig.add_shape(
                            type="rect",
                            x0=stock.index[loc - 1],
                            x1=stock.index[-1],
                            y0=float(stock["Low"].iloc[loc - 1]),
                            y1=float(stock["High"].iloc[loc - 1]),
                            fillcolor="rgba(255, 165, 0, 0.10)",
                            line=dict(width=0),
                            row=1,
                            col=1,
                        )

                sig_names = [
                    "Bullish_Engulfing",
                    "Hammer",
                    "Inverted_Hammer",
                    "Morning_Star",
                    "EMA20_Reclaim",
                    "MACD_Bull_Cross",
                    "RSI_Bounce",
                    "Breakout_5D",
                ]
                for sig in sig_names:
                    y = stock["Low"] * (0.995 if sig in ["Hammer", "Inverted_Hammer"] else 1.005)
                    fig.add_trace(
                        go.Scatter(
                            x=stock.index,
                            y=np.where(stock[sig], y, np.nan),
                            mode="markers",
                            name=sig,
                        ),
                        row=2,
                        col=1,
                    )

                fig.add_trace(go.Scatter(x=stock.index, y=stock["OBV"], name="OBV", mode="lines"), row=2, col=1)

                if show_benchmark_local and not bench.empty:
                    rs_ratio = compute_relative_strength(stock["Close"], bench["Close"])
                    fig.add_trace(go.Scatter(x=rs_ratio.index, y=rs_ratio, name="Stock/Benchmark", mode="lines"), row=3, col=1)
                    fig.add_trace(go.Scatter(x=bench.index, y=bench["Close"], name=f"Benchmark {benchmark_symbol_local}", mode="lines"), row=3, col=1)
                else:
                    fig.add_trace(go.Scatter(x=stock.index, y=stock["RSI14"], name="RSI14", mode="lines"), row=3, col=1)

                fig.add_trace(go.Bar(x=stock.index, y=stock["Volume"], name="Daily Volume"), row=4, col=1)
                fig.add_trace(go.Scatter(x=stock.index, y=stock["VOL_SMA20"], name="Vol SMA20", mode="lines"), row=4, col=1)

                if np.isfinite(float(stock_last["Close"])):
                    fig.add_hline(y=float(stock_last["Close"]), line_width=1.2, line_dash="dash", annotation_text="Current", row=1, col=1)
                if stock_res["decision"] in {"BUY", "STRONG BUY"} and np.isfinite(stock_res["stop_price"]):
                    fig.add_hline(y=float(stock_res["stop_price"]), line_width=1.2, line_dash="dash", annotation_text="Stop", row=1, col=1)
                if stock_res["decision"] in {"BUY", "STRONG BUY"} and np.isfinite(stock_res["entry_price"]):
                    fig.add_hline(y=float(stock_res["entry_price"]), line_width=1.2, line_dash="dash", annotation_text="Entry", row=1, col=1)
                if stock_res["decision"] in {"BUY", "STRONG BUY"} and np.isfinite(stock_res.get("target_1", np.nan)):
                    fig.add_hline(y=float(stock_res["target_1"]), line_width=1.0, line_dash="dot", annotation_text="TP1", row=1, col=1)
                if stock_res["decision"] in {"BUY", "STRONG BUY"} and np.isfinite(stock_res.get("target_2", np.nan)):
                    fig.add_hline(y=float(stock_res["target_2"]), line_width=1.0, line_dash="dot", annotation_text="TP2", row=1, col=1)

                fig.update_layout(height=980, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
                st.subheader("Entry Plan & Risk")
                plan_cols = st.columns(4)
                plan = stock_res.get("entry_plan", {})
                plan_cols[0].metric(
                    "Entry",
                    f"Rp {plan.get('entry_price_plan', np.nan):,.0f}" if pd.notna(plan.get("entry_price_plan", np.nan)) else "n/a",
                    f"Zone {plan.get('entry_zone_low', np.nan):,.0f} - {plan.get('entry_zone_high', np.nan):,.0f}" if pd.notna(plan.get("entry_zone_low", np.nan)) and pd.notna(plan.get("entry_zone_high", np.nan)) else "Zone n/a",
                )
                plan_cols[1].metric(
                    "Stop Loss",
                    f"Rp {plan.get('stop_loss_plan', np.nan):,.0f}" if pd.notna(plan.get("stop_loss_plan", np.nan)) else "n/a",
                    f"Risk / sh. Rp {plan.get('risk_per_share', np.nan):,.0f}" if pd.notna(plan.get("risk_per_share", np.nan)) else "Risk n/a",
                )
                plan_cols[2].metric(
                    "Target 1",
                    f"Rp {plan.get('target_1', np.nan):,.0f}" if pd.notna(plan.get("target_1", np.nan)) else "n/a",
                    f"RR {plan.get('risk_reward_1', np.nan):.2f}" if pd.notna(plan.get('risk_reward_1', np.nan)) else "RR n/a",
                )
                plan_cols[3].metric(
                    "Target 2",
                    f"Rp {plan.get('target_2', np.nan):,.0f}" if pd.notna(plan.get("target_2", np.nan)) else "n/a",
                    f"RR {plan.get('risk_reward_2', np.nan):.2f}" if pd.notna(plan.get('risk_reward_2', np.nan)) else "RR n/a",
                )
                plan_table = pd.DataFrame(
                    [
                        {"Metric": "Plan Reason", "Value": plan.get("plan_reason", "n/a")},
                        {"Metric": "Entry Zone Low", "Value": f"Rp {plan.get('entry_zone_low', np.nan):,.0f}" if pd.notna(plan.get("entry_zone_low", np.nan)) else "n/a"},
                        {"Metric": "Entry Zone High", "Value": f"Rp {plan.get('entry_zone_high', np.nan):,.0f}" if pd.notna(plan.get("entry_zone_high", np.nan)) else "n/a"},
                        {"Metric": "Entry Price", "Value": f"Rp {plan.get('entry_price_plan', np.nan):,.0f}" if pd.notna(plan.get("entry_price_plan", np.nan)) else "n/a"},
                        {"Metric": "Stop Loss", "Value": f"Rp {plan.get('stop_loss_plan', np.nan):,.0f}" if pd.notna(plan.get("stop_loss_plan", np.nan)) else "n/a"},
                        {"Metric": "Target 1", "Value": f"Rp {plan.get('target_1', np.nan):,.0f}" if pd.notna(plan.get("target_1", np.nan)) else "n/a"},
                        {"Metric": "Target 2", "Value": f"Rp {plan.get('target_2', np.nan):,.0f}" if pd.notna(plan.get("target_2", np.nan)) else "n/a"},
                        {"Metric": "RR 1", "Value": f"{plan.get('risk_reward_1', np.nan):.2f}" if pd.notna(plan.get("risk_reward_1", np.nan)) else "n/a"},
                        {"Metric": "RR 2", "Value": f"{plan.get('risk_reward_2', np.nan):.2f}" if pd.notna(plan.get("risk_reward_2", np.nan)) else "n/a"},
                        {"Metric": "Upside to TP1", "Value": f"{plan.get('upside_to_t1_pct', np.nan):.2f}%" if pd.notna(plan.get("upside_to_t1_pct", np.nan)) else "n/a"},
                        {"Metric": "Upside to TP2", "Value": f"{plan.get('upside_to_t2_pct', np.nan):.2f}%" if pd.notna(plan.get("upside_to_t2_pct", np.nan)) else "n/a"},
                    ]
                )
                st.dataframe(plan_table, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.subheader("Detail indikator")
                detail_cols = st.columns(3)
                detail_cols[0].write(f"**Score:** `{stock_res['score']:.2f}`")
                detail_cols[0].write(f"**Core score:** `{stock_res['core_score']:.2f}`")
                detail_cols[0].write(f"**Smart money score:** `{stock_res['smart_money_score']:.2f}`")
                detail_cols[0].write(f"**Decision:** `{stock_res['decision']}`")
                detail_cols[0].write(f"**Dominant cycle:** `{stock_res['dominant_period']} bars`")
                detail_cols[0].write(f"**Time to top / bottom:** `{stock_res.get('time_to_top', np.nan)} / {stock_res['time_to_bottom']} bars`")
                detail_cols[1].write(f"**FVG:** `{stock_res['fvg_present']}`")
                detail_cols[1].write(f"**Order Block:** `{stock_res['ob_present']}`")
                detail_cols[1].write(f"**Reversal score:** `{stock_res['reversal_score']}`")
                detail_cols[1].write(f"**Phase:** `{stock_res['phase']}`")
                detail_cols[1].write(f"**Phase age:** `{stock_res.get('phase_age_bars', np.nan)} bars` ({stock_res.get('phase_age_pct', np.nan):.0f}%)")
                detail_cols[1].write(f"**Cycle reliability:** `{stock_res.get('cycle_reliability', np.nan):.0f}%`")
                detail_cols[1].write(f"**Cycle gate:** `{stock_res.get('cycle_gate_reason', 'OK')}`")
                detail_cols[1].write(f"**Macro gate:** `{stock_res.get('macro_gate_reason', 'OK')}`")
                detail_cols[1].write(f"**Macro score:** `{stock_res.get('macro_score', np.nan):.0f}`")
                detail_cols[2].write(f"**Entry:** `{stock_res['entry_price']:.2f}`" if pd.notna(stock_res["entry_price"]) else "**Entry:** n/a")
                detail_cols[2].write(f"**Stoploss:** `{stock_res['stop_price']:.2f}`" if pd.notna(stock_res["stop_price"]) else "**Stoploss:** n/a")
                detail_cols[2].write(f"**OBV trend:** `{stock_res['obv_trend']}`")
                detail_cols[2].write(f"**Phase confidence:** `{stock_res['phase_confidence']:.0f}%`")
                detail_cols[2].write(f"**PEG:** `{fundamental.get('peg_ratio', np.nan):.2f}`" if pd.notna(fundamental.get("peg_ratio", np.nan)) else "**PEG:** n/a")
                detail_cols[2].write(f"**Revenue growth:** `{format_growth_percent(fundamental.get('revenue_growth', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Earnings growth:** `{format_growth_percent(fundamental.get('earnings_growth', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Current fundamental grade:** `{fundamental.get('fundamental_grade', 'n/a')}`")
                detail_cols[2].write(f"**Future fundamental grade:** `{stock_res.get('future_fundamental_grade', 'n/a')}`")
                detail_cols[2].write(f"**Future fundamental score:** `{stock_res.get('future_fundamental_score', np.nan):.2f}`" if pd.notna(stock_res.get("future_fundamental_score", np.nan)) else "**Future fundamental score:** n/a")
                detail_cols[2].write(f"**Future direction:** `{stock_res.get('future_fundamental_direction', 'n/a')}`")
                detail_cols[2].write(f"**Future confidence:** `{stock_res.get('future_fundamental_confidence', np.nan):.0f}%`" if pd.notna(stock_res.get("future_fundamental_confidence", np.nan)) else "**Future confidence:** n/a")
                detail_cols[2].write(f"**Future phase:** `{stock_res.get('future_fundamental_phase', 'Unknown')}`")
                detail_cols[2].write(f"**Future reason:** `{stock_res.get('future_fundamental_reason', 'n/a')}`")
                detail_cols[2].write(f"**Notes:** `{stock_res['notes']}`")
        else:
            st.info("Masukkan ticker lalu klik **Analyze ticker** untuk membuka deep dive.")

    analysis = st.session_state.get("ifs_analysis", {})
    ifs_context = analysis.get("ifs_context", {})
    stock_res = analysis.get("stock_res", {})
    fundamental = analysis.get("fundamental", {})
    future_context = analysis.get("future_context", {})
    stock_df = analysis.get("stock_df", pd.DataFrame())
    bench_df = analysis.get("bench_df", pd.DataFrame())
    selected_symbol = analysis.get("symbol", normalize_ticker(ticker_input))

    with sub_overview:
        st.subheader("Overview Ranking")
        valid_results = st.session_state.get("global_valid_results", [])
        if valid_results:
            rows = []
            for r in valid_results:
                ifs_score = _safe_float(r.get("ifs_score"), np.nan)
                rows.append(
                    {
                        "Rank": 0,
                        "Ticker": r.get("symbol", "-"),
                        "IFS": round(ifs_score, 2) if pd.notna(ifs_score) else np.nan,
                        "Grade": r.get("ifs_grade", "n/a"),
                        "Forward": round(_safe_float(r.get("ifs_breakdown", {}).get("Forward Fundamental"), np.nan), 1) if isinstance(r.get("ifs_breakdown", {}), dict) else np.nan,
                        "Accum": round(_safe_float(r.get("ifs_breakdown", {}).get("Accumulation"), np.nan), 1) if isinstance(r.get("ifs_breakdown", {}), dict) else np.nan,
                        "RS": round(_safe_float(r.get("ifs_breakdown", {}).get("Relative Strength"), np.nan), 1) if isinstance(r.get("ifs_breakdown", {}), dict) else np.nan,
                        "Qual": round(_safe_float(r.get("ifs_breakdown", {}).get("Quality"), np.nan), 1) if isinstance(r.get("ifs_breakdown", {}), dict) else np.nan,
                        "Catalyst": round(_safe_float(r.get("ifs_breakdown", {}).get("Catalyst"), np.nan), 1) if isinstance(r.get("ifs_breakdown", {}), dict) else np.nan,
                        "Decision": r.get("decision", "-"),
                        "MarketStruct": round(_safe_float(r.get("market_structure_score"), np.nan), 1) if pd.notna(r.get("market_structure_score", np.nan)) else np.nan,
                    }
                )
            ov_df = pd.DataFrame(rows).sort_values(["IFS", "MarketStruct"], ascending=[False, False], na_position="last").reset_index(drop=True)
            ov_df["Rank"] = np.arange(1, len(ov_df) + 1)
            st.dataframe(ov_df.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("Jalankan global scan terlebih dahulu agar ranking IFS muncul di sini.")

    with sub_factor:
        st.subheader(f"Factor Breakdown — {selected_symbol}")
        if ifs_context:
            factor_df = pd.DataFrame(
                [
                    {"Factor": k, "Score": round(v, 2)}
                    for k, v in ifs_context.get("ifs_breakdown", {}).items()
                ]
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("IFS", f'{ifs_context.get("ifs_score", np.nan):.2f}', ifs_context.get("ifs_grade", "n/a"))
            c2.metric("Forward Direction", ifs_context.get("ifs_detail", {}).get("future_direction", "n/a"))
            c3.metric("Confidence", f'{ifs_context.get("ifs_detail", {}).get("future_confidence", np.nan):.0f}%')
            st.dataframe(factor_df, use_container_width=True, hide_index=True)
        else:
            st.info("Klik Analyze ticker untuk melihat breakdown faktor IFS.")

    with sub_smart:
        st.subheader("Smart Money")
        if ifs_context is not None and stock_res is not None:
            sm_cols = st.columns(4)
            sm_cols[0].metric("Smart Money Score", f'{ifs_context.get("ifs_detail", {}).get("smart_money_score", np.nan):.2f}')
            sm_cols[1].metric("Accumulation", f'{ifs_context.get("ifs_detail", {}).get("accumulation_score", np.nan):.2f}')
            sm_cols[2].metric("CMF20", f'{stock_res.get("cmf20", np.nan):.2f}' if pd.notna(stock_res.get("cmf20", np.nan)) else "n/a")
            sm_cols[3].metric("REL_VOL", f'{stock_res.get("rel_vol", np.nan):.2f}' if pd.notna(stock_res.get("rel_vol", np.nan)) else "n/a")
            smart_table = pd.DataFrame(
                [
                    {"Metric": "OBV Trend", "Value": stock_res.get("obv_trend", "n/a")},
                    {"Metric": "OBV Slope", "Value": f'{stock_res.get("obv_slope10", np.nan):.2f}' if pd.notna(stock_res.get("obv_slope10", np.nan)) else "n/a"},
                    {"Metric": "CMF20", "Value": f'{stock_res.get("cmf20", np.nan):.2f}' if pd.notna(stock_res.get("cmf20", np.nan)) else "n/a"},
                    {"Metric": "MFI14", "Value": f'{stock_res.get("mfi14", np.nan):.2f}' if pd.notna(stock_res.get("mfi14", np.nan)) else "n/a"},
                    {"Metric": "Smart Money Score", "Value": f'{ifs_context.get("ifs_detail", {}).get("smart_money_score", np.nan):.2f}'},
                    {"Metric": "Accumulation Score", "Value": f'{ifs_context.get("ifs_detail", {}).get("accumulation_score", np.nan):.2f}'},
                ]
            )
            st.dataframe(smart_table, use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada hasil analisis untuk Smart Money.")

    with sub_forward:
        st.subheader("Forward Fundamental")
        if ifs_context and fundamental:
            cols = st.columns(4)
            cols[0].metric("Current Fundamental", f'{fundamental.get("fundamental_score", np.nan):.2f}', fundamental.get("fundamental_grade", "n/a"))
            cols[1].metric("Future Fundamental", f'{future_context.get("future_fundamental_score", np.nan):.2f}', future_context.get("future_fundamental_grade", "n/a"))
            cols[2].metric("Direction", future_context.get("future_fundamental_direction", "n/a"))
            cols[3].metric("Confidence", f'{future_context.get("future_fundamental_confidence", np.nan):.0f}%')
            forward_table = pd.DataFrame(
                [
                    {"Metric": "Revenue Growth", "Value": format_growth_percent(fundamental.get("revenue_growth", np.nan), decimals=0)},
                    {"Metric": "Earnings Growth", "Value": format_growth_percent(fundamental.get("earnings_growth", np.nan), decimals=0)},
                    {"Metric": "PEG", "Value": f'{fundamental.get("peg_ratio", np.nan):.2f}' if pd.notna(fundamental.get("peg_ratio", np.nan)) else "n/a"},
                    {"Metric": "Future Phase", "Value": future_context.get("future_phase", "Unknown")},
                    {"Metric": "Future Reason", "Value": future_context.get("future_moat_reason", "n/a")},
                    {"Metric": "Future Macro Gate", "Value": future_context.get("future_macro_gate_reason", "OK")},
                ]
            )
            st.dataframe(forward_table, use_container_width=True, hide_index=True)
        else:
            st.info("Klik Analyze ticker untuk melihat forward fundamental.")
