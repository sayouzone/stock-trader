"""
스윙 트레이딩 전략
- 수일~수주 단기 가격 흐름 포착
- RSI 과매도 반전, MA 풀백, 캔들스틱 패턴
- 승률 목표: 50~60%, 손익비: 2:1~3:1
"""

from datetime import datetime

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from indicators.technical import TechnicalIndicators


class SwingTradingStrategy(BaseStrategy):
    """스윙 트레이딩 (Swing Trading)"""

    DEFAULT_PARAMS = {
        "account_risk_pct": 0.0075,    # 0.5~1% per trade
        "max_positions": 5,
        "max_weight": 0.20,
        "atr_stop_multiplier": 1.75,   # ATR 1.5~2
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "risk_reward_target": 2.0,     # 최소 2R
        "time_stop_days": 8,           # 시간 손절 (5~10 거래일)
        "ma_pullback": 20,             # 풀백 기준 MA
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__("스윙 트레이딩", merged)

    def generate_signals(
        self,
        date: datetime,
        historical_data: dict[str, pd.DataFrame],
        portfolio,
        current_prices: dict[str, float],
    ) -> list[dict]:
        signals = []

        # 매도 시그널
        for ticker in list(portfolio.positions.keys()):
            if ticker not in historical_data:
                continue
            df = self.get_indicators(ticker, historical_data[ticker])
            if len(df) < 20:
                continue
            sell_signal = self._check_sell(df, ticker, portfolio, date)
            if sell_signal:
                signals.append(sell_signal)

        # 매수 시그널
        if len(portfolio.positions) >= self.params["max_positions"]:
            return signals

        for ticker, df_raw in historical_data.items():
            if ticker in portfolio.positions:
                continue
            if len(df_raw) < 50:
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

    def _is_uptrend(self, df: pd.DataFrame) -> bool:
        """상승추세 확인: 20MA > 50MA"""
        last = df.iloc[-1]
        sma_20 = last.get("SMA_20")
        sma_50 = last.get("SMA_50")
        if sma_20 and sma_50 and not pd.isna(sma_20) and not pd.isna(sma_50):
            return sma_20 > sma_50
        return False

    def _check_buy(
        self, df: pd.DataFrame, ticker: str, portfolio, prices: dict
    ) -> dict | None:
        """매수: 추세 내 되돌림 반등 or RSI 과매도 반전"""
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        price = prices[ticker]

        entry = False
        reason = ""

        # 조건 1: RSI 과매도 → 복귀 반전
        rsi = last.get("RSI")
        prev_rsi = prev.get("RSI")
        if (
            rsi is not None and prev_rsi is not None
            and not pd.isna(rsi) and not pd.isna(prev_rsi)
        ):
            if prev_rsi <= self.params["rsi_oversold"] and rsi > self.params["rsi_oversold"]:
                entry = True
                reason = f"RSI 과매도 반전 ({prev_rsi:.0f}→{rsi:.0f})"

        # 조건 2: 상승추세 중 MA 풀백 반등
        if not entry and self._is_uptrend(df):
            ma_key = f"SMA_{self.params['ma_pullback']}"
            sma = last.get(ma_key)
            if sma and not pd.isna(sma):
                if prev["Close"] <= sma * 1.01 and last["Close"] > sma:
                    entry = True
                    reason = f"{self.params['ma_pullback']}MA 풀백 반등"

        # 조건 3: 스토캐스틱 과매도 상향 교차
        if not entry:
            k = last.get("Stoch_K")
            d = last.get("Stoch_D")
            prev_k = prev.get("Stoch_K")
            prev_d = prev.get("Stoch_D")
            if all(
                v is not None and not pd.isna(v)
                for v in [k, d, prev_k, prev_d]
            ):
                if prev_k < 20 and prev_k < prev_d and k > d:
                    entry = True
                    reason = "스토캐스틱 과매도 상향교차"

        # 조건 4: 볼린저 밴드 하단 반등 (상승추세 중)
        if not entry and self._is_uptrend(df):
            bb_lower = last.get("BB_Lower")
            if bb_lower and not pd.isna(bb_lower):
                if prev["Low"] <= bb_lower and last["Close"] > bb_lower:
                    entry = True
                    reason = "볼린저 하단 반등"

        if not entry:
            return None

        # 거래량 확인: 반등일 거래량 > 직전 하락일
        if len(df) > 2:
            if last.get("Volume", 0) < prev.get("Volume", 0) * 0.8:
                return None

        # 포지션 사이징 (ATR 기반 손절)
        atr = last.get("ATR", price * 0.02)
        if pd.isna(atr):
            atr = price * 0.02

        stop_loss = price - self.params["atr_stop_multiplier"] * atr
        shares = portfolio.position_size(
            self.params["account_risk_pct"], price, stop_loss
        )

        equity = portfolio.total_equity(prices)
        max_shares = int(equity * self.params["max_weight"] / price)
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

    def _check_sell(
        self, df: pd.DataFrame, ticker: str, portfolio, date: datetime
    ) -> dict | None:
        """매도: 목표 도달, 과매수, 시간 손절"""
        pos = portfolio.positions.get(ticker)
        if not pos:
            return None

        last = df.iloc[-1]
        price = last["Close"]
        atr = last.get("ATR", price * 0.02)
        if pd.isna(atr):
            atr = price * 0.02

        risk_per_share = pos.avg_price - (pos.stop_loss or pos.avg_price * 0.95)
        profit_per_share = price - pos.avg_price

        # 2R 목표 도달
        if risk_per_share > 0:
            r_multiple = profit_per_share / risk_per_share
            if r_multiple >= self.params["risk_reward_target"]:
                return {
                    "ticker": ticker,
                    "action": "SELL",
                    "reason": f"{r_multiple:.1f}R 목표 도달",
                }

        # RSI 과매수
        rsi = last.get("RSI")
        if rsi and not pd.isna(rsi) and rsi >= self.params["rsi_overbought"]:
            if profit_per_share > 0:
                return {
                    "ticker": ticker,
                    "action": "SELL",
                    "reason": f"RSI 과매수 ({rsi:.0f})",
                }

        # 볼린저 밴드 상단 도달
        bb_upper = last.get("BB_Upper")
        if bb_upper and not pd.isna(bb_upper):
            if price >= bb_upper and profit_per_share > 0:
                return {
                    "ticker": ticker,
                    "action": "SELL",
                    "reason": "볼린저 상단 도달",
                }

        # 시간 손절
        holding_days = (date - pos.entry_date).days
        if holding_days >= self.params["time_stop_days"]:
            if profit_per_share <= 0:
                return {
                    "ticker": ticker,
                    "action": "SELL",
                    "reason": f"시간 손절 ({holding_days}일)",
                }

        # ATR 트레일링 스톱 업데이트
        if pos.stop_loss:
            new_stop = price - self.params["atr_stop_multiplier"] * atr
            if new_stop > pos.stop_loss:
                pos.stop_loss = new_stop

        return None
