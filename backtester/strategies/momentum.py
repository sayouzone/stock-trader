"""
모멘텀 트레이딩 전략
- 듀얼 모멘텀 (절대 + 상대)
- 정기 리밸런싱, 모멘텀 점수 기반 종목 교체
- 승률 목표: 45~55%, 손익비: 2:1~3:1
"""

from datetime import datetime

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from indicators.technical import TechnicalIndicators


class MomentumTradingStrategy(BaseStrategy):
    """모멘텀 트레이딩 (Momentum Trading)"""

    DEFAULT_PARAMS = {
        "account_risk_pct": 0.015,
        "max_positions": 15,
        "max_weight": 0.10,
        "rebalance_freq_days": 21,   # 월간 리밸런싱
        "momentum_weights": {        # 복합 모멘텀 가중
            "1m": 0.3,
            "3m": 0.3,
            "6m": 0.2,
            "12m": 0.2,
        },
        "top_pct": 0.20,            # 상위 20% 선정
        "abs_momentum_period": 252,  # 12개월 절대 모멘텀
        "adx_threshold": 25,
        "stop_loss_pct": 0.10,      # -10% 개별 손절
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__("모멘텀 트레이딩", merged)
        self._last_rebalance = None
        self._rebalance_count = 0

    def generate_signals(
        self,
        date: datetime,
        historical_data: dict[str, pd.DataFrame],
        portfolio,
        current_prices: dict[str, float],
    ) -> list[dict]:
        signals = []

        # 리밸런싱 주기 확인
        if self._last_rebalance is not None:
            days_since = (date - self._last_rebalance).days
            if days_since < self.params["rebalance_freq_days"]:
                # 리밸런싱 기간 아님 → 손절만 체크
                for ticker in list(portfolio.positions.keys()):
                    if ticker in current_prices:
                        pos = portfolio.positions[ticker]
                        pnl_pct = pos.unrealized_pnl_pct(current_prices[ticker])
                        if pnl_pct <= -self.params["stop_loss_pct"] * 100:
                            signals.append({
                                "ticker": ticker,
                                "action": "SELL",
                                "reason": f"손절 ({pnl_pct:.1f}%)",
                            })
                return signals

        # ── 리밸런싱 실행 ──

        # 1. 모멘텀 점수 계산
        scores = self._compute_momentum_scores(historical_data)
        if not scores:
            return signals

        # 2. 듀얼 모멘텀 필터링
        qualified = self._apply_dual_momentum(scores, historical_data)

        # 3. 상위 N개 선정
        n_select = max(1, int(len(qualified) * self.params["top_pct"]))
        n_select = min(n_select, self.params["max_positions"])
        selected = sorted(qualified.items(), key=lambda x: x[1], reverse=True)[:n_select]
        selected_tickers = set(t for t, _ in selected)

        # 4. 기존 보유 중 순위 밖 → 매도
        for ticker in list(portfolio.positions.keys()):
            if ticker not in selected_tickers:
                signals.append({
                    "ticker": ticker,
                    "action": "SELL",
                    "reason": "모멘텀 순위 하락 → 리밸런싱 매도",
                })

        # 5. 신규 편입 → 매수
        for ticker, score in selected:
            if ticker in portfolio.positions:
                continue
            if ticker not in current_prices:
                continue

            price = current_prices[ticker]
            stop_loss = price * (1 - self.params["stop_loss_pct"])

            # 균등 배분
            equity = portfolio.total_equity(current_prices)
            target_value = equity * self.params["max_weight"]
            shares = int(target_value / price)

            if shares > 0:
                signals.append({
                    "ticker": ticker,
                    "action": "BUY",
                    "shares": shares,
                    "stop_loss": stop_loss,
                    "reason": f"모멘텀 상위 편입 (점수: {score:.2f})",
                })

        self._last_rebalance = date
        self._rebalance_count += 1

        return signals

    def _compute_momentum_scores(
        self, historical_data: dict[str, pd.DataFrame]
    ) -> dict[str, float]:
        """복합 모멘텀 점수 계산"""
        scores = {}
        weights = self.params["momentum_weights"]

        for ticker, df in historical_data.items():
            if len(df) < 252:
                continue

            close = df["Close"]
            try:
                ret_1m = close.iloc[-1] / close.iloc[-21] - 1 if len(df) > 21 else 0
                ret_3m = close.iloc[-1] / close.iloc[-63] - 1 if len(df) > 63 else 0
                ret_6m = close.iloc[-1] / close.iloc[-126] - 1 if len(df) > 126 else 0
                ret_12m = close.iloc[-1] / close.iloc[-252] - 1 if len(df) > 252 else 0

                # skip-month: 직전 1개월 제외 (단기 반전 효과 제거)
                score = (
                    weights["1m"] * ret_1m
                    + weights["3m"] * ret_3m
                    + weights["6m"] * ret_6m
                    + weights["12m"] * ret_12m
                )
                scores[ticker] = score
            except (IndexError, ZeroDivisionError):
                continue

        return scores

    def _apply_dual_momentum(
        self,
        scores: dict[str, float],
        historical_data: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """듀얼 모멘텀: 절대 모멘텀 양 + 상대 모멘텀 상위"""
        qualified = {}

        for ticker, score in scores.items():
            df = historical_data.get(ticker)
            if df is None or len(df) < self.params["abs_momentum_period"]:
                continue

            # 절대 모멘텀: 12개월 수익률 > 0
            close = df["Close"]
            ret_12m = close.iloc[-1] / close.iloc[-252] - 1
            if ret_12m <= 0:
                continue

            # 200MA 위 확인
            sma_200 = close.rolling(200).mean().iloc[-1]
            if pd.isna(sma_200) or close.iloc[-1] < sma_200:
                continue

            qualified[ticker] = score

        return qualified
