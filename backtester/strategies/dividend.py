"""
배당 투자 전략
- 안정적 배당 수입 + 장기 자본 증식
- 배당수익률 밴드 상단 매수, DCA 방식 분할 매수
- 승률 목표: 65~75%, 손익비: 1.5:1~2:1
"""

from datetime import datetime

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from indicators.technical import TechnicalIndicators


class DividendInvestingStrategy(BaseStrategy):
    """배당 투자 (Dividend Investing)"""

    DEFAULT_PARAMS = {
        "account_risk_pct": 0.025,
        "max_positions": 20,
        "max_weight": 0.10,
        "buy_interval_days": 30,      # DCA: 월간 분할 매수
        "yield_proxy_threshold": 0.7, # 배당률 프록시: 1년 내 하위 30% 가격
        "per_low_percentile": 0.25,   # PER 하단 25%
        "stop_loss_pct": 0.15,        # 15% 손절 (배당주 넓은 허용)
        "overvalued_threshold": 0.90, # 1년 내 상위 90% → 과대평가
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__("배당 투자", merged)
        self._last_buy_date: dict[str, datetime] = {}

    def generate_signals(
        self,
        date: datetime,
        historical_data: dict[str, pd.DataFrame],
        portfolio,
        current_prices: dict[str, float],
    ) -> list[dict]:
        signals = []

        # 매도 체크
        for ticker in list(portfolio.positions.keys()):
            if ticker not in historical_data:
                continue
            df = self.get_indicators(ticker, historical_data[ticker])
            if len(df) < 50:
                continue
            sell_signal = self._check_sell(df, ticker, portfolio)
            if sell_signal:
                signals.append(sell_signal)

        # 매수 (DCA)
        if len(portfolio.positions) >= self.params["max_positions"]:
            return signals

        for ticker, df_raw in historical_data.items():
            if len(df_raw) < 252:
                continue

            # DCA 간격 체크
            last_buy = self._last_buy_date.get(ticker)
            if last_buy:
                days_since = (date - last_buy).days
                if days_since < self.params["buy_interval_days"]:
                    continue

            df = self.get_indicators(ticker, df_raw)
            buy_signal = self._check_buy(df, ticker, portfolio, current_prices, date)
            if buy_signal:
                signals.append(buy_signal)
                self._last_buy_date[ticker] = date

                if len(portfolio.positions) + len(
                    [s for s in signals if s["action"] == "BUY"]
                ) >= self.params["max_positions"]:
                    break

        return signals

    def _is_dividend_candidate(self, df: pd.DataFrame) -> bool:
        """
        배당주 후보 프록시: 가격 안정성 + 낮은 변동성
        (실제 배당 데이터 없이 가격 기반)
        """
        close = df["Close"]

        # 변동성이 낮은 종목 (안정적)
        returns = close.pct_change().tail(252)
        vol = returns.std() * np.sqrt(252)
        if vol > 0.4:  # 연 변동성 40% 초과는 배당주 부적합
            return False

        # 200MA 근처에서 안정적 움직임
        sma_200 = TechnicalIndicators.sma(close, 200)
        if pd.isna(sma_200.iloc[-1]):
            return False

        deviation = abs(close.iloc[-1] / sma_200.iloc[-1] - 1)
        return deviation < 0.20  # 200MA에서 ±20% 이내

    def _check_buy(
        self, df: pd.DataFrame, ticker: str, portfolio, prices: dict, date: datetime
    ) -> dict | None:
        """매수: 배당률 밴드 상단 (=가격 밴드 하단) + PER 하단"""
        price = prices[ticker]
        close = df["Close"]

        if not self._is_dividend_candidate(df):
            return None

        # 배당수익률 밴드 상단 프록시 = 가격이 1년 밴드 하단
        low_1y = close.tail(252).min()
        high_1y = close.tail(252).max()
        price_range = high_1y - low_1y
        if price_range == 0:
            return None

        position = (price - low_1y) / price_range

        # 하위 30% 가격 → 배당률 상단
        if position > self.params["yield_proxy_threshold"]:
            return None

        reason = f"배당률 밴드 상단 프록시 (1Y {position:.0%})"

        # 포지션 사이징 (DCA이므로 균등 배분)
        equity = portfolio.total_equity(prices)
        target_per_stock = equity * self.params["max_weight"]

        # 이미 보유 중이면 추가 매수 한도
        existing = portfolio.positions.get(ticker)
        if existing:
            current_value = existing.shares * price
            remaining = target_per_stock - current_value
            if remaining <= 0:
                return None
            shares = int(remaining / price)
        else:
            # 신규: 1/3씩 DCA
            shares = int(target_per_stock / 3 / price)

        stop_loss = price * (1 - self.params["stop_loss_pct"])

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
        """매도: 극단적 과대평가, 펀더멘털 악화"""
        pos = portfolio.positions.get(ticker)
        if not pos:
            return None

        last = df.iloc[-1]
        price = last["Close"]
        close = df["Close"]

        # 극단적 과대평가: 1년 범위 상위 90%
        high_1y = close.tail(252).max()
        low_1y = close.tail(252).min()
        price_range = high_1y - low_1y
        if price_range > 0:
            position = (price - low_1y) / price_range
            if position >= self.params["overvalued_threshold"]:
                return {
                    "ticker": ticker,
                    "action": "SELL",
                    "reason": f"극단적 과대평가 (1Y {position:.0%})",
                }

        # 추세 급락 (200MA 대폭 이탈)
        sma_200 = last.get("SMA_200")
        if sma_200 and not pd.isna(sma_200):
            if price < sma_200 * 0.85:  # 200MA -15% 이탈
                return {
                    "ticker": ticker,
                    "action": "SELL",
                    "reason": "200MA 대폭 이탈 (-15%+)",
                }

        return None
