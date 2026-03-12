"""
포지션 트레이딩 전략
- 수주~수개월 단위 중기 추세 추종
- 200MA 기반 시장 판단, Stage 2 분석, ATR 손절
- 승률 목표: 40~50%, 손익비: 3:1 이상
"""

from datetime import datetime

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from indicators.technical import TechnicalIndicators


class PositionTradingStrategy(BaseStrategy):
    """포지션 트레이딩 (Position Trading)"""

    DEFAULT_PARAMS = {
        "account_risk_pct": 0.015,    # 1.5% per trade
        "max_positions": 8,
        "max_weight": 0.20,
        "atr_stop_multiplier": 2.5,   # ATR 2~3배 손절
        "ma_fast": 20,
        "ma_slow": 50,
        "ma_trend": 200,
        "adx_threshold": 25,
        "volume_breakout_ratio": 1.5,  # 150% 거래량
        "trailing_ma": 50,             # 트레일링 스톱 기준 MA
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__("포지션 트레이딩", merged)

    def generate_signals(
        self,
        date: datetime,
        historical_data: dict[str, pd.DataFrame],
        portfolio,
        current_prices: dict[str, float],
    ) -> list[dict]:
        signals = []

        # 매도 시그널 먼저 처리
        for ticker in list(portfolio.positions.keys()):
            if ticker not in historical_data:
                continue
            df = self.get_indicators(ticker, historical_data[ticker])
            if len(df) < self.params["ma_trend"]:
                continue

            sell_signal = self._check_sell(df, ticker, portfolio)
            if sell_signal:
                signals.append(sell_signal)

        # 매수 시그널
        if len(portfolio.positions) >= self.params["max_positions"]:
            return signals

        for ticker, df_raw in historical_data.items():
            if ticker in portfolio.positions:
                continue
            if len(df_raw) < self.params["ma_trend"]:
                continue

            df = self.get_indicators(ticker, df_raw)
            buy_signal = self._check_buy(df, ticker, portfolio, current_prices)
            if buy_signal:
                signals.append(buy_signal)

                if len(portfolio.positions) + len(
                    [s for s in signals if s["action"] == "BUY"]
                ) >= self.params["max_positions"]:
                    break

        return signals

    def _check_market_condition(self, df: pd.DataFrame) -> bool:
        """STEP 1: 시장 환경 판단 - 200MA 위인가?"""
        if f"SMA_{self.params['ma_trend']}" not in df.columns:
            return False
        last = df.iloc[-1]
        return last["Close"] > last[f"SMA_{self.params['ma_trend']}"]

    def _check_buy(
        self, df: pd.DataFrame, ticker: str, portfolio, prices: dict
    ) -> dict | None:
        """STEP 3~4: 종목 선정 + 매수 타이밍"""
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        # 시장 환경: 200MA 위
        if not self._check_market_condition(df):
            return None

        # ADX > 25 (추세 확인)
        if pd.isna(last.get("ADX")) or last["ADX"] < self.params["adx_threshold"]:
            return None

        # Stage 2: 주가가 50MA 위, 50MA 상승 중
        sma_50 = last.get(f"SMA_{self.params['ma_slow']}")
        if pd.isna(sma_50) or last["Close"] < sma_50:
            return None

        # 진입 조건: MA 풀백 반등 또는 돌파
        entry = False
        reason = ""

        # 조건 1: 20일 MA 풀백 후 반등
        sma_fast = last.get(f"SMA_{self.params['ma_fast']}")
        prev_close = prev["Close"]
        if sma_fast and not pd.isna(sma_fast):
            if prev_close <= sma_fast * 1.02 and last["Close"] > sma_fast:
                entry = True
                reason = f"20MA 풀백 반등"

        # 조건 2: MACD 시그널 교차
        if not entry:
            macd_h = last.get("MACD_Hist")
            prev_macd_h = prev.get("MACD_Hist")
            if (
                macd_h is not None
                and prev_macd_h is not None
                and not pd.isna(macd_h)
                and not pd.isna(prev_macd_h)
            ):
                if prev_macd_h < 0 and macd_h > 0:
                    entry = True
                    reason = "MACD 시그널 상향 교차"

        # 조건 3: 볼린저 밴드 Squeeze 후 확장
        if not entry:
            bb_width = last.get("BB_Width")
            if bb_width and not pd.isna(bb_width):
                recent_bw = df["BB_Width"].tail(20)
                if bb_width > recent_bw.quantile(0.3) and recent_bw.min() < recent_bw.quantile(0.1):
                    if last["Close"] > last.get("BB_Middle", 0):
                        entry = True
                        reason = "볼린저 Squeeze 돌파"

        if not entry:
            return None

        # STEP 5: 포지션 사이징 (ATR 기반 손절)
        price = prices[ticker]
        atr = last.get("ATR", price * 0.03)
        if pd.isna(atr):
            atr = price * 0.03

        stop_loss = price - self.params["atr_stop_multiplier"] * atr
        shares = portfolio.position_size(
            self.params["account_risk_pct"], price, stop_loss
        )

        # 최대 비중 체크
        equity = portfolio.total_equity(prices)
        max_invest = equity * self.params["max_weight"]
        max_shares = int(max_invest / price)
        shares = min(shares, max_shares)

        if shares <= 0:
            return None

        return {
            "ticker": ticker,
            "action": "BUY",
            "shares": shares,
            "stop_loss": stop_loss,
            "reason": reason,
        }

    def _check_sell(self, df: pd.DataFrame, ticker: str, portfolio) -> dict | None:
        """STEP 7: 매도 조건"""
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        # 50일 MA 하향 이탈 (종가 기준)
        sma_trail = last.get(f"SMA_{self.params['trailing_ma']}")
        if sma_trail and not pd.isna(sma_trail):
            if last["Close"] < sma_trail:
                return {
                    "ticker": ticker,
                    "action": "SELL",
                    "reason": f"{self.params['trailing_ma']}MA 하향 이탈",
                }

        # RSI/MACD 다이버전스
        rsi = last.get("RSI")
        macd_h = last.get("MACD_Hist")
        prev_macd_h = prev.get("MACD_Hist")
        if (
            macd_h is not None
            and prev_macd_h is not None
            and not pd.isna(macd_h)
            and not pd.isna(prev_macd_h)
        ):
            if prev_macd_h > 0 and macd_h < 0:
                if rsi and not pd.isna(rsi) and rsi > 70:
                    return {
                        "ticker": ticker,
                        "action": "SELL",
                        "reason": "MACD 하향 + RSI 과매수",
                    }

        # ADX 하락 전환
        adx = last.get("ADX")
        if len(df) > 5:
            prev_adx = df["ADX"].iloc[-5]
            if adx and prev_adx and not pd.isna(adx) and not pd.isna(prev_adx):
                if adx < prev_adx and adx < 20:
                    return {
                        "ticker": ticker,
                        "action": "SELL",
                        "reason": "ADX 하락 전환 (추세 약화)",
                    }

        # 트레일링 스톱 업데이트 (ATR 기반)
        pos = portfolio.positions.get(ticker)
        if pos:
            atr = last.get("ATR", 0)
            if not pd.isna(atr) and atr > 0:
                new_stop = last["Close"] - self.params["atr_stop_multiplier"] * atr
                if pos.stop_loss and new_stop > pos.stop_loss:
                    pos.stop_loss = new_stop

        return None
