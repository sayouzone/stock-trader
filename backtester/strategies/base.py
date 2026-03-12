"""
전략 기본 클래스 (추상)
모든 투자 전략은 이 클래스를 상속받아 구현
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd

from indicators.technical import TechnicalIndicators


class BaseStrategy(ABC):
    """투자 전략 추상 기본 클래스"""

    def __init__(self, name: str, params: Optional[dict] = None):
        self.name = name
        self.params = params or {}
        self.indicators = TechnicalIndicators()
        self._indicator_cache: dict[str, pd.DataFrame] = {}

    def initialize(self, data: dict[str, pd.DataFrame]) -> None:
        """전략 초기화 - 지표 사전 계산"""
        for ticker, df in data.items():
            self._indicator_cache[ticker] = TechnicalIndicators.compute_all(df)

    def get_indicators(self, ticker: str, data: pd.DataFrame) -> pd.DataFrame:
        """캐시된 지표 반환 또는 새로 계산"""
        if ticker in self._indicator_cache:
            cache = self._indicator_cache[ticker]
            if len(cache) >= len(data):
                return cache.loc[data.index]
        result = TechnicalIndicators.compute_all(data)
        self._indicator_cache[ticker] = result
        return result

    @abstractmethod
    def generate_signals(
        self,
        date: datetime,
        historical_data: dict[str, pd.DataFrame],
        portfolio,
        current_prices: dict[str, float],
    ) -> list[dict]:
        """
        시그널 생성 (각 전략에서 구현)

        Returns:
            list of signal dicts:
            {
                "ticker": str,
                "action": "BUY" | "SELL",
                "shares": int,
                "stop_loss": float (optional),
                "reason": str,
            }
        """
        pass

    def _account_risk_pct(self) -> float:
        """1회 최대 리스크 비율"""
        return self.params.get("account_risk_pct", 0.01)

    def _max_positions(self) -> int:
        """최대 동시 보유 종목 수"""
        return self.params.get("max_positions", 10)

    def _max_weight(self) -> float:
        """한 종목 최대 비중"""
        return self.params.get("max_weight", 0.20)
