
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
# Tab 1: Global Watchlist Top 20 + reversal signals
# Tab 2: Deep Dive per ticker + benchmark / bandarmology / backtest / time analysis
# =========================================================

st.set_page_config(page_title="IDX Dual Tab Scanner", layout="wide")
st.title("📊 IDX Dual Tab Scanner")
st.caption(
    "Global watchlist untuk ranking cepat, lalu deep dive untuk bedah detail per ticker dengan time analysis, bandarmology, dan backtest."
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

    try:
        info = yf.Ticker(base).info or {}
    except Exception:
        info = {}

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

    try:
        info = yf.Ticker(base).info or {}
    except Exception:
        info = {}

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
        }

    series = close.astype(float).copy()
    log_close = np.log(series.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    basis = log_close if len(log_close) >= 30 else series

    min_period = 5
    max_period = int(min(120, max(20, n // 2)))
    max_period = max(min_period + 1, max_period)

    smooth_window = int(np.clip(n // 20, 3, 9))
    trend_window = int(np.clip(n // 4, 10, max(15, max_period)))

    smooth = basis.rolling(window=smooth_window, center=True, min_periods=1).mean()
    detrended = smooth - smooth.rolling(window=trend_window, center=True, min_periods=1).mean()
    detrended = detrended.bfill().ffill()
    arr = detrended.to_numpy(dtype=float)
    arr = arr - np.nanmean(arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def confidence_from_peak(peak: float, baseline: float) -> float:
        if not np.isfinite(peak):
            return 0.0
        base = abs(baseline) + 1e-9
        return float(np.clip(peak / base, 0.0, 10.0))

    fft_period = np.nan
    fft_conf = 0.0
    frequencies, power = periodogram(arr)
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
        analytic = hilbert(arr)
        phase = np.unwrap(np.angle(analytic))
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
    x = arr - np.mean(arr)
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

    order = int(np.clip(n // 30, 2, 10))
    minima = argrelextrema(series.values, np.less_equal, order=order)[0]
    if len(minima) > 0:
        recent_cutoff = max(0, n - max_period * 2)
        recent_minima = minima[minima >= recent_cutoff]
        anchor_idx = int(recent_minima[-1] if len(recent_minima) else minima[-1])
    else:
        window = min(max_period, n)
        anchor_idx = int(max(0, n - window))

    bars_since_anchor = max(0, (n - 1) - anchor_idx)
    rem = bars_since_anchor % dominant_period
    time_to_next_bottom = 0 if rem == 0 else dominant_period - rem
    threshold = max(4, int(round(dominant_period * 0.15)))
    cycle_ok = (time_to_next_bottom <= threshold) or (bars_since_anchor <= threshold)

    composite_conf = float(np.clip(np.nanmean([fft_conf / 3.0, hilbert_conf, autocorr_conf]), 0.0, 1.0) * 100)
    details = {
        "fft_period": int(fft_period) if np.isfinite(fft_period) else np.nan,
        "hilbert_period": int(hilbert_period) if np.isfinite(hilbert_period) else np.nan,
        "autocorr_period": int(autocorr_period) if np.isfinite(autocorr_period) else np.nan,
        "weighted_period": int(dominant_period),
        "fft_confidence": float(np.clip(fft_conf * 10, 0.0, 100.0)),
        "hilbert_confidence": float(np.clip(hilbert_conf * 100, 0.0, 100.0)),
        "autocorr_confidence": float(np.clip(autocorr_conf * 100, 0.0, 100.0)),
        "composite_confidence": composite_conf,
        "anchor_idx": int(anchor_idx),
        "bars_since_anchor": int(bars_since_anchor),
        "threshold": int(threshold),
    }

    return dominant_period, int(time_to_next_bottom), cycle_ok, details

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

    liquidity_ok = (d["Volume"].tail(20).mean() >= min_avg_volume) and (min_price <= float(last["Close"]) <= max_price)
    trend_ok = (last["Close"] > last["EMA20"]) and (last["EMA50"] > last["EMA200"])
    smc_confirmed = d["Bullish_FVG"].tail(8).any() or d["Bullish_OB"].tail(8).any()

    if mode == "Conservative":
        buy_threshold, strong_threshold = 80, 88
    elif mode == "Balanced":
        buy_threshold, strong_threshold = 70, 83
    else:
        buy_threshold, strong_threshold = 60, 74

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
    if reversal_score == 0:
        notes.append("Belum_Ada_Reversal_Strong")

    return {
        "valid": True,
        "symbol": None,
        "decision": decision,
        "score": float(final_score),
        "core_score": float(core_score),
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
        "cycle_info": cycle_info,
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


# =========================================================
# Backtest Engine
# =========================================================
def backtest_system_smc(
    d: pd.DataFrame,
    buy_threshold: float,
    stop_loss_atr: float,
    take_profit_atr: float,
    hold_days: int,
    cost_bps: float,
) -> pd.DataFrame:
    d = d.copy()
    if d.empty or len(d) < 50:
        return pd.DataFrame()

    scores = []
    signals = []
    core_max = (12 * 4) + (4 * 3) + (4 * 2) + (4 * 3)

    for i in range(len(d)):
        row = d.iloc[i]
        prev_row = d.iloc[i - 1] if i > 0 else row

        fvg_tail = d["Bullish_FVG"].iloc[max(0, i - 5) : i + 1].any()
        ob_tail = d["Bullish_OB"].iloc[max(0, i - 5) : i + 1].any()
        ema20_reclaim = d["EMA20_Reclaim"].iloc[max(0, i - 5) : i + 1].any()
        macd_cross = d["MACD_Bull_Cross"].iloc[max(0, i - 5) : i + 1].any()
        rsi_bounce = d["RSI_Bounce"].iloc[max(0, i - 5) : i + 1].any()
        breakout_5d = d["Breakout_5D"].iloc[max(0, i - 5) : i + 1].any()

        smc_pts = (
            (4 * int(fvg_tail))
            + (4 * int(ob_tail))
            + (2 * int(row["REL_VOL"] >= 1.25))
            + (2 * int(row["VPT"] > d["VPT"].iloc[max(i - 5, 0)]))
        )
        trend_pts = (
            int(row["Close"] > row["EMA20"])
            + int(row["EMA20"] > row["EMA50"])
            + int(row["EMA50"] > row["EMA200"])
            + int(row["EMA50"] > prev_row["EMA50"])
        )
        mom_pts = (
            int(50 <= row["RSI14"] <= 72)
            + int(row["MACD_HIST"] > 0)
            + int(row["Close"] > row["BB_MID"])
            + int(row["ADX14"] >= 18)
        )
        rev_pts = int(ema20_reclaim) + int(macd_cross) + int(rsi_bounce) + int(breakout_5d)

        score = ((smc_pts * 4) + (trend_pts * 3) + (mom_pts * 2) + (rev_pts * 3)) / core_max * 100
        scores.append(score)

        sig = (
            (score >= buy_threshold)
            and (row["Close"] > row["EMA20"])
            and (row["EMA50"] > row["EMA200"])
            and (fvg_tail or ob_tail or ema20_reclaim or macd_cross or breakout_5d)
        )
        signals.append(sig)

    d["SCORE"] = scores
    d["SIGNAL"] = signals

    trades = []
    i = 0
    n = len(d)
    cost = cost_bps / 10000.0

    while i < n - 2:
        if not bool(d.iloc[i]["SIGNAL"]):
            i += 1
            continue

        entry_i = i + 1
        if entry_i >= n:
            break

        entry_price = float(d.iloc[entry_i]["Open"])
        atr_val = float(d.iloc[i]["ATR14"])
        if not np.isfinite(atr_val) or atr_val <= 0:
            i += 1
            continue

        stop_price = entry_price - stop_loss_atr * atr_val
        target_price = entry_price + take_profit_atr * atr_val

        exit_price = None
        exit_date = None
        exit_reason = None
        end_i = min(entry_i + hold_days, n - 1)
        last_j = entry_i

        for j in range(entry_i, end_i + 1):
            last_j = j
            day = d.iloc[j]

            if float(day["Low"]) <= stop_price:
                exit_price = stop_price
                exit_date = d.index[j]
                exit_reason = "Stop Loss (ATR)"
                break

            if float(day["High"]) >= target_price:
                exit_price = target_price
                exit_date = d.index[j]
                exit_reason = "Target Profit (ATR)"
                break

            if j >= entry_i + 2 and float(day["Close"]) < float(day["EMA20"]) and float(day["MACD_HIST"]) < 0:
                exit_price = float(day["Close"])
                exit_date = d.index[j]
                exit_reason = "Trend Broken"
                break

            if j == end_i:
                exit_price = float(day["Close"])
                exit_date = d.index[j]
                exit_reason = "Time Exhausted"
                break

        if exit_price is None:
            i += 1
            continue

        gross_ret = (exit_price / entry_price) - 1
        net_ret = gross_ret - (2 * cost)

        trades.append(
            {
                "Entry Date": d.index[entry_i],
                "Exit Date": exit_date,
                "Entry": entry_price,
                "Exit": exit_price,
                "Reason": exit_reason,
                "Return %": net_ret * 100,
                "Holding Bars": int(last_j - entry_i),
            }
        )
        i = max(entry_i + 1, last_j + 1)

    return pd.DataFrame(trades)


def build_stats(trades_df: pd.DataFrame):
    if trades_df.empty:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "avg_ret": np.nan,
            "profit_factor": np.nan,
            "max_dd": np.nan,
        }

    rets = trades_df["Return %"] / 100.0
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    pf = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else np.inf
    equity = (1 + rets.fillna(0)).cumprod()

    return {
        "trades": int(len(trades_df)),
        "win_rate": float((rets > 0).mean() * 100),
        "avg_ret": float(rets.mean() * 100),
        "profit_factor": float(pf),
        "max_dd": float(max_drawdown(equity) * 100),
    }


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


def process_symbol(symbol: str):
    try:
        d = load_ticker_data(symbol, months)
        if d.empty or len(d) < min_history_bars:
            return {"valid": False, "symbol": symbol, "reason": "Data historis tidak mencukupi"}

        res = score_stock_smc(
            d,
            flow_used=False,
            flow_val=50,
            min_avg_volume=min_avg_volume,
            min_price=min_price,
            max_price=max_price,
            mode=GLOBAL_MODE,
            min_history_bars=min_history_bars,
        )
        res["symbol"] = symbol
        return res
    except Exception as e:
        return {"valid": False, "symbol": symbol, "reason": str(e)}


# =========================================================
# Tabs
# =========================================================
tab1, tab2 = st.tabs(["📈 Global Watchlist", "🔎 Deep Dive Analysis"])

with tab1:
    st.subheader("Global Watchlist Top 20")
    st.caption("Ranking berdasarkan score gabungan struktur, reversal, SMC, dan momentum.")

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
                            "Reversal_Score": r["reversal_score"],
                            "Close": round(r["close"], 2),
                            "RSI14": round(r["rsi"], 2),
                            "ADX14": round(r["adx"], 2) if pd.notna(r["adx"]) else np.nan,
                            "RelVol": round(r["rel_vol"], 2) if pd.notna(r["rel_vol"]) else np.nan,
                            "OBV": r["obv_trend"],
                            "FVG": "🔥 YES" if r["fvg_present"] else "NO",
                            "OrderBlock": "🎯 YES" if r["ob_present"] else "NO",
                            "Trend": "BULLISH" if r["trend_ok"] else "BEARISH",
                            "Reversal": r["reversal_hits"],
                            "Phase": r.get("phase", "-"),
                            "PhaseConf": round(r.get("phase_confidence", np.nan), 0) if pd.notna(r.get("phase_confidence", np.nan)) else np.nan,
                            "Cycle": r.get("dominant_period", np.nan),
                            "CycleTTB": r.get("time_to_bottom", np.nan),
                            "SmartMoney": round(r.get("smart_money_score", np.nan), 2) if pd.notna(r.get("smart_money_score", np.nan)) else np.nan,
                            "Entry": round(r["entry_price"], 2) if pd.notna(r["entry_price"]) else np.nan,
                            "Stop": round(r["stop_price"], 2) if pd.notna(r["stop_price"]) else np.nan,
                            "Notes": r["notes"],
                        }
                    )

                watch_df = (
                    pd.DataFrame(watch_rows)
                    .sort_values(["Score", "SmartMoney", "Reversal_Score", "RelVol"], ascending=[False, False, False, False], na_position="last")
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
                                value=f"Rp {row.Close:,.0f}",
                                delta=f"Score: {row.Score}",
                            )
                            st.markdown(
                                f"**Reversal:** `{row.Reversal}`  \n"
                                f"**FVG/OB:** `{row.FVG}` / `{row.OrderBlock}`  \n"
                                f"**OBV:** `{row.OBV}`  \n"
                                f"**Entry:** `{row.Entry}`  \n"
                                f"**Stop:** `{row.Stop}`"
                            )
                else:
                    st.info("Belum ada kandidat BUY/STRONG BUY pada universe saat ini.")

                st.markdown("---")
                st.subheader("🏆 Global Watchlist Ranking (Top 20)")
                st.dataframe(top20, use_container_width=True, hide_index=True)
    else:
        if not st.session_state.global_watch_df.empty:
            st.subheader("🏆 Global Watchlist Ranking (Top 20)")
            st.dataframe(st.session_state.global_watch_df.head(20), use_container_width=True, hide_index=True)
            st.info("Klik **Run global scan** di sidebar untuk memperbarui ranking.")
        else:
            st.info("Klik **Run global scan** di sidebar untuk mulai scan universe.")

with tab2:
    st.subheader("🔎 Deep Dive Analysis")
    st.caption("Bandarmology, strategy mode, backtest, benchmark IHSG, dan time analysis tersedia di sini.")

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
            hold_days_local = st.slider("Maksimum hold (hari bursa)", 5, 60, 15, key="deep_hold_days")
        with d4:
            trade_cost_bps_local = st.slider("Biaya transaksi (bps/side)", 0, 50, 10, key="deep_trade_cost_bps")

        d5, d6 = st.columns([1, 1])
        with d5:
            take_profit_atr_local = st.slider("Take Profit (x ATR)", 1.0, 5.0, 2.5, 0.1, key="deep_take_profit_atr")
        with d6:
            stop_loss_atr_local = st.slider("Stop Loss (x ATR)", 1.0, 5.0, 1.8, 0.1, key="deep_stop_loss_atr")

        analyze_btn = st.button("Analyze ticker", type="primary", key="deep_analyze_btn")

    if analyze_btn:
        deep_ticker = normalize_ticker(ticker_input)
        flow_val_local = map_flow_to_score(bandarmology_mode)

        stock_df = load_ticker_data(deep_ticker, months)
        bench_df = load_ticker_data(benchmark_symbol_local, months) if benchmark_symbol_local else pd.DataFrame()

        if stock_df.empty or len(stock_df) < min_history_bars:
            st.error("Data ticker tidak cukup atau gagal diunduh.")
        else:
            stock_res = score_stock_smc(
                stock_df,
                flow_used=True,
                flow_val=flow_val_local,
                min_avg_volume=min_avg_volume,
                min_price=min_price,
                max_price=max_price,
                mode=strategy_mode,
                min_history_bars=min_history_bars,
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
            bench = pd.DataFrame()
            bench_cycle = None
            if show_benchmark_local and not bench_df.empty and len(bench_df) >= min_history_bars:
                bench = bench_df.copy()
                bench_cycle = compute_cycle_features(bench["Close"])

            stock_status = "Near Bottom" if stock_res["time_to_bottom"] <= 4 else "Mid-Cycle Moving"
            bench_status = "n/a"
            if bench_cycle is not None:
                bench_status = "Near Bottom" if bench_cycle[1] <= 4 else "Mid-Cycle Moving"

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
                    <div>• <b>Status Posisi Siklus:</b> {stock_status}</div>
                    <div>• <b>8-Phase Cycle:</b> {stock_res["phase"]} ({stock_res["phase_confidence"]:.0f}%)</div>
                    <div>• <b>FFT / Hilbert / Autocorr:</b> {stock_cycle_info.get("fft_period", "-")} / {stock_cycle_info.get("hilbert_period", "-")} / {stock_cycle_info.get("autocorr_period", "-")}</div>
                    <div>• <b>Weighted Composite:</b> {stock_cycle_info.get("weighted_period", stock_period)} bars</div>
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
                st.write(f"**Time to next bottom:** `{stock_res['time_to_bottom']} bars`")
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

            fig.update_layout(height=980, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            left2, right2 = st.columns([1, 1])
            with left2:
                st.subheader("Performance / Backtest")
                bt_threshold = {"Conservative": 80, "Balanced": 70, "Aggressive": 60}[strategy_mode]
                trades = backtest_system_smc(
                    stock,
                    bt_threshold,
                    stop_loss_atr_local,
                    take_profit_atr_local,
                    hold_days_local,
                    trade_cost_bps_local,
                )
                stats = build_stats(trades)
                stats_df = pd.DataFrame(
                    [
                        ["Total Executed Trades", stats["trades"]],
                        ["Win Rate %", None if pd.isna(stats["win_rate"]) else f"{stats['win_rate']:.2f}%"],
                        ["Avg Return %", None if pd.isna(stats["avg_ret"]) else f"{stats['avg_ret']:.2f}%"],
                        ["Profit Factor", None if pd.isna(stats["profit_factor"]) else round(stats["profit_factor"], 2)],
                        ["Max Drawdown %", None if pd.isna(stats["max_dd"]) else f"{stats['max_dd']:.2f}%"],
                    ],
                    columns=["Metric", "Value"],
                )
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

            with right2:
                st.subheader("Time Analysis - Benchmark")
                if show_benchmark_local and not bench.empty and bench_cycle is not None:
                    st.write(f"**Benchmark:** `{benchmark_symbol_local}`")
                    st.write(f"**Dominant cycle:** `{bench_cycle[0]} bars`")
                    st.write(f"**Time to next bottom:** `{bench_cycle[1]} bars`")
                    st.write(f"**Cycle status:** `{'Near Bottom' if bench_cycle[1] <= 4 else 'Mid-Cycle Moving'}`")
                else:
                    st.info("Benchmark chart dimatikan atau data benchmark tidak tersedia.")

                if not trades.empty:
                    st.caption("10 trade terakhir dari backtest:")
                    st.dataframe(trades.tail(10), use_container_width=True, hide_index=True)
                else:
                    st.info("Belum ada trade terbentuk dari aturan backtest saat ini.")

            st.markdown("---")
            st.subheader("Detail indikator")
            detail_cols = st.columns(3)
            detail_cols[0].write(f"**Score:** `{stock_res['score']:.2f}`")
            detail_cols[0].write(f"**Core score:** `{stock_res['core_score']:.2f}`")
            detail_cols[0].write(f"**Smart money score:** `{stock_res['smart_money_score']:.2f}`")
            detail_cols[0].write(f"**Decision:** `{stock_res['decision']}`")
            detail_cols[0].write(f"**Dominant cycle:** `{stock_res['dominant_period']} bars`")
            detail_cols[1].write(f"**FVG:** `{stock_res['fvg_present']}`")
            detail_cols[1].write(f"**Order Block:** `{stock_res['ob_present']}`")
            detail_cols[1].write(f"**Reversal score:** `{stock_res['reversal_score']}`")
            detail_cols[1].write(f"**Phase:** `{stock_res['phase']}`")
            detail_cols[2].write(f"**Entry:** `{stock_res['entry_price']:.2f}`" if pd.notna(stock_res["entry_price"]) else "**Entry:** n/a")
            detail_cols[2].write(f"**Stoploss:** `{stock_res['stop_price']:.2f}`" if pd.notna(stock_res["stop_price"]) else "**Stoploss:** n/a")
            detail_cols[2].write(f"**OBV trend:** `{stock_res['obv_trend']}`")
            detail_cols[2].write(f"**Phase confidence:** `{stock_res['phase_confidence']:.0f}%`")
            detail_cols[2].write(f"**PEG:** `{fundamental.get('peg_ratio', np.nan):.2f}`" if pd.notna(fundamental.get("peg_ratio", np.nan)) else "**PEG:** n/a")
            detail_cols[2].write(f"**Fundamental grade:** `{fundamental.get('fundamental_grade', 'n/a')}`")
            detail_cols[2].write(f"**Notes:** `{stock_res['notes']}`")
    else:
        st.info("Masukkan ticker lalu klik **Analyze ticker** untuk membuka deep dive.")
