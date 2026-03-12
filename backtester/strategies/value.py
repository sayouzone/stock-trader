"""
가치 트레이딩 전략
- 내재가치 대비 저평가 종목 발굴 → 리레이팅 시 매도
- 골든크로스, MACD 0선 돌파 진입
- 승률 목표: 55~65%, 손익비: 2:1~3:1
"""

from datetime import datetime

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from indicators.technical import TechnicalIndicators


class ValueTradingStrategy(BaseStrategy):
    """가치 트레이딩 (Value Trading)"""

    DEFAULT_PARAMS = {
        "account_risk_pct": 0.025,
        "max_positions": 15,
        "max_weight": 0.15,
        "lookback_low": 252,          # 1년 저점 비교
        "recovery_threshold": 0.10,   # 저점 대비 10% 반등 시 진입 검토
        "golden_cross_fast": 50,
        "golden_cross_slow": 200,
        "exit_high_percentile": 0.85, # 1년 내 상위 85% 도달 시 매도 시작
        "holding_check_days": 60,     # 60일 이상 횡보 시 재검토
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__("가치 트레이딩", merged)

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
            if len(df) < 50:
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
            if len(df_raw) < self.params["lookback_low"]:
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

    def _is_undervalued_proxy(self, df: pd.DataFrame) -> bool:
        """
        저평가 프록시: 가격이 1년 저점 근처에 있고 반등 시작
        (실제 PER/PBR 데이터 없이 가격 기반 대리 지표)
        """
        lookback = min(self.params["lookback_low"], len(df))
        recent = df.tail(lookback)
        low_1y = recent["Low"].min()
        high_1y = recent["High"].max()
        current = df["Close"].iloc[-1]

        price_range = high_1y - low_1y
        if price_range == 0:
            return False

        # 현재가가 1년 범위 하위 30% 이내
        position_in_range = (current - low_1y) / price_range
        return position_in_range < 0.30

    def _check_buy(
        self, df: pd.DataFrame, ticker: str, portfolio, prices: dict
    ) -> dict | None:
        """매수 조건: 저평가 + 기술적 반전 신호"""
        last = df.iloc[-1]
        price = prices[ticker]

        # 저평가 프록시 확인
        if not self._is_undervalued_proxy(df):
            return None

        entry = False
        reason = ""

        # 조건 1: 골든크로스 (50MA > 200MA 상향 돌파)
        gc = last.get("GoldenCross_50_200")
        if gc and not pd.isna(gc) and gc == 1:
            entry = True
            reason = "골든크로스 (50/200)"

        # 조건 2: MACD 0선 돌파
        if not entry:
            macd = last.get("MACD")
            prev_macd = df["MACD"].iloc[-2] if len(df) > 1 else None
            if (
                macd is not None and prev_macd is not None
                and not pd.isna(macd) and not pd.isna(prev_macd)
            ):
                if prev_macd < 0 and macd > 0:
                    entry = True
                    reason = "MACD 0선 상향 돌파"

        # 조건 3: 하락추세 둔화 + 거래량 급증
        if not entry:
            if len(df) > 20:
                recent_vol = df["Volume"].tail(5).mean()
                avg_vol = df["Volume"].tail(60).mean()
                if avg_vol > 0 and recent_vol / avg_vol > 2.0:
                    # 저점이 더 이상 낮아지지 않는 패턴
                    lows_10 = df["Low"].tail(10)
                    lows_20 = df["Low"].tail(20).head(10)
                    if lows_10.min() >= lows_20.min() * 0.98:
                        entry = True
                        reason = "하락 둔화 + 거래량 급증"

        if not entry:
            return None

        # 포지션 사이징: 1년 저점 아래 손절
        lookback = min(self.params["lookback_low"], len(df))
        low_1y = df["Low"].tail(lookback).min()
        stop_loss = low_1y * 0.97  # 1년 저점 -3%

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

    def _check_sell(self, df: pd.DataFrame, ticker: str, portfolio) -> dict | None:
        """매도: 가치 회복 완료 (리레이팅)"""
        pos = portfolio.positions.get(ticker)
        if not pos:
            return None

        last = df.iloc[-1]
        price = last["Close"]
        pnl_pct = pos.unrealized_pnl_pct(price)

        # 1년 범위에서 상위 85% 도달 → 내재가치 도달 프록시
        lookback = min(self.params["lookback_low"], len(df))
        recent = df.tail(lookback)
        high_1y = recent["High"].max()
        low_1y = recent["Low"].min()
        price_range = high_1y - low_1y

        if price_range > 0:
            position = (price - low_1y) / price_range
            if position >= self.params["exit_high_percentile"]:
                return {
                    "ticker": ticker,
                    "action": "SELL",
                    "reason": f"밸류에이션 리레이팅 완료 (1Y {position:.0%})",
                }

        # 데드크로스
        gc = last.get("GoldenCross_50_200")
        if gc and not pd.isna(gc) and gc == -1:
            return {
                "ticker": ticker,
                "action": "SELL",
                "reason": "데드크로스 (50/200)",
            }

        # 장기 횡보 (시간 손절)
        holding_days = (df.index[-1] - pos.entry_date).days
        if holding_days > self.params["holding_check_days"]:
            if abs(pnl_pct) < 3:  # 60일 이상 보유 + ±3% 이내 횡보
                return {
                    "ticker": ticker,
                    "action": "SELL",
                    "reason": f"{holding_days}일 보유 횡보 → 기회비용",
                }

        return None
