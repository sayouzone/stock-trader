"""
기술적 지표 계산 모듈
- 문서에 언급된 모든 기술적 지표를 한 DataFrame에 추가
- 각 스크리너에서 공통으로 사용
"""

import numpy as np
import pandas as pd


def sma(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(p, min_periods=p).mean()


def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()


def rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1 / p, min_periods=p).mean()
    l = (-d).clip(lower=0).ewm(alpha=1 / p, min_periods=p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def macd(s: pd.Series, fast=12, slow=26, sig=9):
    ml = ema(s, fast) - ema(s, slow)
    sl = ema(ml, sig)
    return ml, sl, ml - sl


def bollinger(s: pd.Series, p=20, sd=2.0):
    m = sma(s, p)
    st = s.rolling(p).std()
    return m + sd * st, m, m - sd * st


def atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / p, min_periods=p).mean()


def adx(df: pd.DataFrame, p: int = 14) -> pd.Series:
    plus_dm = df["High"].diff().clip(lower=0)
    minus_dm = (-df["Low"].diff()).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
    _atr = atr(df, p)
    pdi = 100 * plus_dm.ewm(alpha=1 / p, min_periods=p).mean() / _atr
    mdi = 100 * minus_dm.ewm(alpha=1 / p, min_periods=p).mean() / _atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / p, min_periods=p).mean()


def stochastic(df: pd.DataFrame, k_p=14, d_p=3):
    lo = df["Low"].rolling(k_p).min()
    hi = df["High"].rolling(k_p).max()
    k = 100 * (df["Close"] - lo) / (hi - lo).replace(0, np.nan)
    return k, k.rolling(d_p).mean()


def volume_ratio(vol: pd.Series, p: int = 50) -> pd.Series:
    return vol / vol.rolling(p).mean().replace(0, np.nan)


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame에 주요 기술적 지표 일괄 추가"""
    r = df.copy()
    c = r["Close"]

    for p in [5, 10, 20, 50, 150, 200]:
        r[f"SMA_{p}"] = sma(c, p)
    for p in [10, 21]:
        r[f"EMA_{p}"] = ema(c, p)

    r["RSI"] = rsi(c)

    ml, sl, hist = macd(c)
    r["MACD"] = ml
    r["MACD_Signal"] = sl
    r["MACD_Hist"] = hist

    bu, bm, bl = bollinger(c)
    r["BB_Upper"] = bu
    r["BB_Middle"] = bm
    r["BB_Lower"] = bl
    r["BB_Width"] = (bu - bl) / bm

    r["ATR"] = atr(r)
    r["ADX"] = adx(r)

    sk, sd = stochastic(r)
    r["Stoch_K"] = sk
    r["Stoch_D"] = sd

    if "Volume" in r.columns:
        r["Vol_Ratio"] = volume_ratio(r["Volume"])

    # 수익률
    for p in [5, 21, 63, 126, 252]:
        r[f"Return_{p}d"] = c.pct_change(p)

    # 52주 고점/저점
    r["High_52w"] = c.rolling(252, min_periods=50).max()
    r["Low_52w"] = c.rolling(252, min_periods=50).min()
    r["Pct_from_52H"] = (c - r["High_52w"]) / r["High_52w"]
    r["Pct_from_52L"] = (c - r["Low_52w"]) / r["Low_52w"]

    return r
