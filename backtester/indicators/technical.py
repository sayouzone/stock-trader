"""
기술적 지표 계산 모듈
- 문서에 언급된 모든 기술적 지표 구현
- 이동평균, RSI, MACD, 볼린저밴드, ATR, ADX, 스토캐스틱, 파라볼릭SAR 등
"""

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """기술적 지표 계산 클래스"""

    # ── 이동평균 ──

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """단순 이동평균선 (SMA)"""
        return series.rolling(window=period, min_periods=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """지수 이동평균선 (EMA)"""
        return series.ewm(span=period, adjust=False).mean()

    # ── RSI ──

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """상대강도지수 (RSI)"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    # ── MACD ──

    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (선, 시그널, 히스토그램)"""
        fast_ema = series.ewm(span=fast, adjust=False).mean()
        slow_ema = series.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    # ── 볼린저 밴드 ──

    @staticmethod
    def bollinger_bands(
        series: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """볼린저 밴드 (상단, 중간, 하단)"""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower

    @staticmethod
    def bollinger_bandwidth(
        series: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> pd.Series:
        """볼린저 밴드 폭 (Squeeze 감지용)"""
        upper, middle, lower = TechnicalIndicators.bollinger_bands(
            series, period, std_dev
        )
        return (upper - lower) / middle

    # ── ATR ──

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        high = df["High"]
        low = df["Low"]
        close = df["Close"].shift(1)

        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.ewm(alpha=1 / period, min_periods=period).mean()

    # ── ADX ──

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index (추세 강도)"""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        atr = TechnicalIndicators.atr(df, period)

        plus_di = 100 * (
            plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr
        )
        minus_di = 100 * (
            minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr
        )

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1 / period, min_periods=period).mean()
        return adx

    # ── 스토캐스틱 ──

    @staticmethod
    def stochastic(
        df: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> tuple[pd.Series, pd.Series]:
        """스토캐스틱 %K, %D"""
        low_min = df["Low"].rolling(window=k_period).min()
        high_max = df["High"].rolling(window=k_period).max()

        k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
        d = k.rolling(window=d_period).mean()
        return k, d

    # ── 피보나치 되돌림 ──

    @staticmethod
    def fibonacci_levels(
        high: float, low: float
    ) -> dict[str, float]:
        """피보나치 되돌림 수준"""
        diff = high - low
        return {
            "0.0%": high,
            "23.6%": high - diff * 0.236,
            "38.2%": high - diff * 0.382,
            "50.0%": high - diff * 0.5,
            "61.8%": high - diff * 0.618,
            "100.0%": low,
        }

    # ── 상대강도 (RS) ──

    @staticmethod
    def relative_strength(
        stock: pd.Series, benchmark: pd.Series, period: int = 252
    ) -> pd.Series:
        """상대강도: 종목 수익률 / 벤치마크 수익률"""
        stock_ret = stock.pct_change(period)
        bench_ret = benchmark.pct_change(period)
        return stock_ret / bench_ret.replace(0, np.nan)

    @staticmethod
    def rs_rank(
        returns: dict[str, float],
    ) -> dict[str, float]:
        """RS 순위 (백분위) 계산"""
        if not returns:
            return {}
        sorted_items = sorted(returns.items(), key=lambda x: x[1])
        n = len(sorted_items)
        return {
            ticker: (rank + 1) / n * 100
            for rank, (ticker, _) in enumerate(sorted_items)
        }

    # ── 골든크로스 / 데드크로스 ──

    @staticmethod
    def ma_crossover(
        series: pd.Series, fast_period: int, slow_period: int
    ) -> pd.Series:
        """이동평균 교차 시그널: 1=골든크로스, -1=데드크로스, 0=없음"""
        fast_ma = TechnicalIndicators.sma(series, fast_period)
        slow_ma = TechnicalIndicators.sma(series, slow_period)

        cross = pd.Series(0, index=series.index)
        above = fast_ma > slow_ma
        cross[above & ~above.shift(1).fillna(False)] = 1   # 골든크로스
        cross[~above & above.shift(1).fillna(True)] = -1   # 데드크로스
        return cross

    # ── 파라볼릭 SAR ──

    @staticmethod
    def parabolic_sar(
        df: pd.DataFrame,
        af_start: float = 0.02,
        af_step: float = 0.02,
        af_max: float = 0.2,
    ) -> pd.Series:
        """파라볼릭 SAR"""
        high = df["High"].values
        low = df["Low"].values
        close = df["Close"].values
        n = len(close)

        sar = np.zeros(n)
        af = af_start
        uptrend = True
        ep = high[0]
        sar[0] = low[0]

        for i in range(1, n):
            if uptrend:
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
                sar[i] = min(sar[i], low[i - 1])
                if i >= 2:
                    sar[i] = min(sar[i], low[i - 2])

                if low[i] < sar[i]:
                    uptrend = False
                    sar[i] = ep
                    ep = low[i]
                    af = af_start
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_step, af_max)
            else:
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
                sar[i] = max(sar[i], high[i - 1])
                if i >= 2:
                    sar[i] = max(sar[i], high[i - 2])

                if high[i] > sar[i]:
                    uptrend = True
                    sar[i] = ep
                    ep = high[i]
                    af = af_start
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_step, af_max)

        return pd.Series(sar, index=df.index, name="SAR")

    # ── 다이버전스 감지 ──

    @staticmethod
    def detect_divergence(
        price: pd.Series, indicator: pd.Series, lookback: int = 20
    ) -> pd.Series:
        """
        베어리시 다이버전스 감지: 가격 신고가 vs 지표 하락
        1 = 베어리시 다이버전스, 0 = 없음
        """
        result = pd.Series(0, index=price.index)

        for i in range(lookback, len(price)):
            window = slice(i - lookback, i + 1)
            price_window = price.iloc[window]
            ind_window = indicator.iloc[window]

            # 가격 최근 고점이 이전 고점보다 높은데 지표는 낮은 경우
            price_high = price_window.idxmax()
            if price_high == price.index[i]:
                prev_highs = price_window.iloc[:-5]
                if len(prev_highs) > 0:
                    prev_high_idx = prev_highs.idxmax()
                    if (
                        price.loc[price_high] > price.loc[prev_high_idx]
                        and indicator.loc[price_high] < indicator.loc[prev_high_idx]
                    ):
                        result.iloc[i] = 1

        return result

    # ── 거래량 분석 ──

    @staticmethod
    def volume_ratio(volume: pd.Series, period: int = 50) -> pd.Series:
        """거래량 비율 (현재 / 평균)"""
        avg_vol = volume.rolling(window=period).mean()
        return volume / avg_vol.replace(0, np.nan)

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        direction = np.sign(close.diff())
        return (volume * direction).cumsum()

    # ── 52주 신고가/신저가 ──

    @staticmethod
    def near_52week_high(close: pd.Series, threshold: float = 0.95) -> pd.Series:
        """52주 신고가 근접 여부 (기본 95% 이상)"""
        high_52 = close.rolling(window=252, min_periods=50).max()
        return close >= high_52 * threshold

    # ── 종합 지표 계산 ──

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame에 주요 기술적 지표 일괄 추가"""
        result = df.copy()
        close = result["Close"]

        # 이동평균
        for p in [5, 10, 20, 50, 150, 200]:
            result[f"SMA_{p}"] = TechnicalIndicators.sma(close, p)
        for p in [10, 21]:
            result[f"EMA_{p}"] = TechnicalIndicators.ema(close, p)

        # RSI
        result["RSI"] = TechnicalIndicators.rsi(close)

        # MACD
        macd_l, macd_s, macd_h = TechnicalIndicators.macd(close)
        result["MACD"] = macd_l
        result["MACD_Signal"] = macd_s
        result["MACD_Hist"] = macd_h

        # 볼린저 밴드
        bb_u, bb_m, bb_l = TechnicalIndicators.bollinger_bands(close)
        result["BB_Upper"] = bb_u
        result["BB_Middle"] = bb_m
        result["BB_Lower"] = bb_l
        result["BB_Width"] = TechnicalIndicators.bollinger_bandwidth(close)

        # ATR
        result["ATR"] = TechnicalIndicators.atr(result)

        # ADX
        result["ADX"] = TechnicalIndicators.adx(result)

        # 스토캐스틱
        k, d = TechnicalIndicators.stochastic(result)
        result["Stoch_K"] = k
        result["Stoch_D"] = d

        # 파라볼릭 SAR
        result["SAR"] = TechnicalIndicators.parabolic_sar(result)

        # 거래량 비율
        if "Volume" in result.columns:
            result["Vol_Ratio"] = TechnicalIndicators.volume_ratio(result["Volume"])

        # 이동평균 교차
        result["GoldenCross_50_200"] = TechnicalIndicators.ma_crossover(close, 50, 200)

        return result
