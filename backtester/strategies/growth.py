"""
성장주 트레이딩 전략
- CAN SLIM / SEPA 방법론 기반
- 피벗 포인트 돌파 매수, 7~8% 고정 손절
- 승률 목표: 35~45%, 손익비: 3:1~5:1
"""

from datetime import datetime

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from indicators.technical import TechnicalIndicators


class GrowthTradingStrategy(BaseStrategy):
    """성장주 트레이딩 (Growth Trading)"""

    DEFAULT_PARAMS = {
        "account_risk_pct": 0.0125,   # 1~1.5% per trade
        "max_positions": 6,
        "max_weight": 0.25,
        "stop_loss_pct": 0.075,        # 7.5% 고정 손절 (오닐 골든룰)
        "buy_zone_pct": 0.05,          # 피벗 +5% 이내
        "volume_breakout_ratio": 1.5,  # 50일 평균 대비 150%+
        "base_period": 30,             # 베이스 패턴 탐색 기간
        "profit_target_1": 0.20,       # +20% 1차 매도
        "profit_target_2": 0.50,       # +50% 2차 매도
        "trail_ma": 50,                # 50MA 트레일링
    }

    def __init__(self, params: dict = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__("성장주 트레이딩", merged)
        self._profit_taken: dict[str, int] = {}  # 이익 실현 단계 추적

    def generate_signals(
        self,
        date: datetime,
        historical_data: dict[str, pd.DataFrame],
        portfolio,
        current_prices: dict[str, float],
    ) -> list[dict]:
        signals = []

        # 매도 시그널 먼저
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

        candidates = []
        for ticker, df_raw in historical_data.items():
            if ticker in portfolio.positions:
                continue
            if len(df_raw) < 200:
                continue

            df = self.get_indicators(ticker, df_raw)
            score = self._score_candidate(df, ticker)
            if score > 0:
                candidates.append((ticker, df, score))

        # 점수 높은 순서로 매수
        candidates.sort(key=lambda x: x[2], reverse=True)

        for ticker, df, score in candidates:
            buy_signal = self._check_buy(df, ticker, portfolio, current_prices)
            if buy_signal:
                signals.append(buy_signal)
                if len(portfolio.positions) + len(
                    [s for s in signals if s["action"] == "BUY"]
                ) >= self.params["max_positions"]:
                    break

        return signals

    def _score_candidate(self, df: pd.DataFrame, ticker: str) -> float:
        """종목 점수 (RS, 추세 강도 등 기반)"""
        last = df.iloc[-1]
        score = 0.0

        # Stage 2 확인: 150MA 위
        sma_150 = last.get("SMA_150")
        if pd.isna(sma_150) or last["Close"] < sma_150:
            return 0

        # 50MA > 200MA
        sma_50 = last.get("SMA_50")
        sma_200 = last.get("SMA_200")
        if sma_50 and sma_200 and not pd.isna(sma_50) and not pd.isna(sma_200):
            if sma_50 > sma_200:
                score += 1
            else:
                return 0

        # ADX 강도
        adx = last.get("ADX")
        if adx and not pd.isna(adx) and adx > 25:
            score += 1

        # 52주 신고가 근접
        if len(df) >= 252:
            high_52 = df["High"].tail(252).max()
            if last["Close"] >= high_52 * 0.85:
                score += 1
            if last["Close"] >= high_52 * 0.95:
                score += 1

        # 거래량 추세
        vol_ratio = last.get("Vol_Ratio")
        if vol_ratio and not pd.isna(vol_ratio) and vol_ratio > 1.0:
            score += 0.5

        return score

    def _check_buy(
        self, df: pd.DataFrame, ticker: str, portfolio, prices: dict
    ) -> dict | None:
        """피벗 포인트 돌파 매수"""
        if len(df) < self.params["base_period"]:
            return None

        last = df.iloc[-1]
        price = prices[ticker]

        # 베이스 패턴의 저항선 (피벗 포인트)
        base = df.tail(self.params["base_period"])
        pivot = base["High"].max()

        # 피벗 돌파 확인
        prev_close = df["Close"].iloc[-2] if len(df) > 1 else 0
        if not (prev_close < pivot and price >= pivot):
            return None

        # Buy Zone 체크 (+5% 이내)
        if price > pivot * (1 + self.params["buy_zone_pct"]):
            return None

        # 거래량 확인 (150%+)
        vol_ratio = last.get("Vol_Ratio")
        if vol_ratio and not pd.isna(vol_ratio):
            if vol_ratio < self.params["volume_breakout_ratio"]:
                return None

        # 포지션 사이징 (7~8% 고정 손절)
        stop_loss = price * (1 - self.params["stop_loss_pct"])
        shares = portfolio.position_size(
            self.params["account_risk_pct"], price, stop_loss
        )

        # 최대 비중
        equity = portfolio.total_equity(prices)
        max_shares = int(equity * self.params["max_weight"] / price)
        shares = min(shares, max_shares)

        if shares <= 0:
            return None

        self._profit_taken[ticker] = 0

        return {
            "ticker": ticker,
            "action": "BUY",
            "shares": shares,
            "stop_loss": stop_loss,
            "reason": f"피벗 돌파 ({pivot:,.0f})",
        }

    def _check_sell(self, df: pd.DataFrame, ticker: str, portfolio) -> dict | None:
        """매도 조건"""
        pos = portfolio.positions.get(ticker)
        if not pos:
            return None

        last = df.iloc[-1]
        price = last["Close"]
        pnl_pct = pos.unrealized_pnl_pct(price)
        stage = self._profit_taken.get(ticker, 0)

        # 이익 실현 가이드
        if stage == 0 and pnl_pct >= self.params["profit_target_1"] * 100:
            self._profit_taken[ticker] = 1
            sell_shares = pos.shares // 3
            if sell_shares > 0:
                return {
                    "ticker": ticker,
                    "action": "SELL",
                    "shares": sell_shares,
                    "reason": f"+{pnl_pct:.1f}% 1차 이익실현",
                }

        if stage == 1 and pnl_pct >= self.params["profit_target_2"] * 100:
            self._profit_taken[ticker] = 2
            sell_shares = pos.shares // 2
            if sell_shares > 0:
                return {
                    "ticker": ticker,
                    "action": "SELL",
                    "shares": sell_shares,
                    "reason": f"+{pnl_pct:.1f}% 2차 이익실현",
                }

        # 손절선 상향: +5% → 본전, +10% → +5%
        if pnl_pct >= 10:
            new_stop = pos.avg_price * 1.05
            if pos.stop_loss and new_stop > pos.stop_loss:
                pos.stop_loss = new_stop
        elif pnl_pct >= 5:
            new_stop = pos.avg_price
            if pos.stop_loss and new_stop > pos.stop_loss:
                pos.stop_loss = new_stop

        # 50MA 이탈 매도
        sma_50 = last.get("SMA_50")
        if sma_50 and not pd.isna(sma_50):
            if price < sma_50:
                vol_ratio = last.get("Vol_Ratio", 1)
                if not pd.isna(vol_ratio) and vol_ratio > 1.2:
                    return {
                        "ticker": ticker,
                        "action": "SELL",
                        "reason": "50MA 이탈 + 거래량 증가",
                    }

        # MACD 다이버전스
        if len(df) > 20:
            rsi = last.get("RSI")
            macd_h = last.get("MACD_Hist")
            prev_macd = df["MACD_Hist"].iloc[-2]
            if (
                macd_h is not None and prev_macd is not None
                and not pd.isna(macd_h) and not pd.isna(prev_macd)
            ):
                if prev_macd > 0 and macd_h < 0:
                    if rsi and not pd.isna(rsi) and rsi > 65:
                        return {
                            "ticker": ticker,
                            "action": "SELL",
                            "reason": "MACD 하향 + RSI 과매수",
                        }

        return None
