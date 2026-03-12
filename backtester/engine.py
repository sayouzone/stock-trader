"""
백테스팅 엔진 코어 모듈
- BacktestEngine: 전략 실행 및 포트폴리오 시뮬레이션
- Portfolio: 계좌 잔고, 포지션, 거래 이력 관리
- Trade: 개별 거래 기록
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Trade:
    """개별 거래 기록"""

    ticker: str
    action: TradeAction
    date: datetime
    price: float
    shares: int
    strategy: str
    reason: str = ""
    commission: float = 0.0

    @property
    def total_value(self) -> float:
        return self.price * self.shares

    def __repr__(self) -> str:
        return (
            f"Trade({self.action.value} {self.ticker} "
            f"{self.shares}주 @ {self.price:,.0f} on {self.date:%Y-%m-%d} "
            f"[{self.reason}])"
        )


@dataclass
class Position:
    """보유 포지션"""

    ticker: str
    shares: int
    avg_price: float
    entry_date: datetime
    stop_loss: Optional[float] = None
    strategy: str = ""

    @property
    def cost_basis(self) -> float:
        return self.avg_price * self.shares

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.avg_price) * self.shares

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.avg_price == 0:
            return 0.0
        return (current_price - self.avg_price) / self.avg_price * 100


class Portfolio:
    """포트폴리오 (계좌) 관리"""

    def __init__(
        self,
        initial_capital: float = 100_000_000,
        commission_rate: float = 0.00015,
        tax_rate: float = 0.0023,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate  # 매매 수수료율
        self.tax_rate = tax_rate  # 세금 (매도 시)
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[dict] = []
        self.closed_trades: list[dict] = []

    def buy(
        self,
        ticker: str,
        date: datetime,
        price: float,
        shares: int,
        strategy: str = "",
        reason: str = "",
        stop_loss: Optional[float] = None,
    ) -> Optional[Trade]:
        """매수 실행"""
        commission = price * shares * self.commission_rate
        total_cost = price * shares + commission

        if total_cost > self.cash:
            # 매수 가능 수량으로 조정
            shares = int((self.cash - commission) / price)
            if shares <= 0:
                return None
            commission = price * shares * self.commission_rate
            total_cost = price * shares + commission

        self.cash -= total_cost

        # 기존 포지션에 추가 or 신규 포지션
        if ticker in self.positions:
            pos = self.positions[ticker]
            total_shares = pos.shares + shares
            pos.avg_price = (
                (pos.avg_price * pos.shares + price * shares) / total_shares
            )
            pos.shares = total_shares
            if stop_loss is not None:
                pos.stop_loss = stop_loss
        else:
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                avg_price=price,
                entry_date=date,
                stop_loss=stop_loss,
                strategy=strategy,
            )

        trade = Trade(
            ticker=ticker,
            action=TradeAction.BUY,
            date=date,
            price=price,
            shares=shares,
            strategy=strategy,
            reason=reason,
            commission=commission,
        )
        self.trades.append(trade)
        return trade

    def sell(
        self,
        ticker: str,
        date: datetime,
        price: float,
        shares: Optional[int] = None,
        strategy: str = "",
        reason: str = "",
    ) -> Optional[Trade]:
        """매도 실행"""
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        if shares is None or shares >= pos.shares:
            shares = pos.shares

        commission = price * shares * self.commission_rate
        tax = price * shares * self.tax_rate
        proceeds = price * shares - commission - tax

        self.cash += proceeds

        # 매매 결과 기록
        pnl = (price - pos.avg_price) * shares - commission - tax
        pnl_pct = (price - pos.avg_price) / pos.avg_price * 100
        holding_days = (date - pos.entry_date).days

        self.closed_trades.append(
            {
                "ticker": ticker,
                "strategy": pos.strategy,
                "entry_date": pos.entry_date,
                "exit_date": date,
                "entry_price": pos.avg_price,
                "exit_price": price,
                "shares": shares,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "holding_days": holding_days,
                "reason": reason,
            }
        )

        # 포지션 업데이트
        pos.shares -= shares
        if pos.shares <= 0:
            del self.positions[ticker]

        trade = Trade(
            ticker=ticker,
            action=TradeAction.SELL,
            date=date,
            price=price,
            shares=shares,
            strategy=strategy,
            reason=reason,
            commission=commission + tax,
        )
        self.trades.append(trade)
        return trade

    def total_equity(self, prices: dict[str, float]) -> float:
        """총 자산 (현금 + 보유 포지션 평가액)"""
        positions_value = sum(
            pos.shares * prices.get(pos.ticker, pos.avg_price)
            for pos in self.positions.values()
        )
        return self.cash + positions_value

    def record_equity(self, date: datetime, prices: dict[str, float]) -> None:
        """일별 자산 기록"""
        equity = self.total_equity(prices)
        self.equity_curve.append(
            {
                "date": date,
                "equity": equity,
                "cash": self.cash,
                "positions_value": equity - self.cash,
                "num_positions": len(self.positions),
            }
        )

    def position_size(
        self,
        account_risk_pct: float,
        entry_price: float,
        stop_price: float,
    ) -> int:
        """포지션 사이징: 매수수량 = (계좌×리스크%) ÷ (진입가-손절가)"""
        equity = self.cash + sum(
            p.shares * p.avg_price for p in self.positions.values()
        )
        risk_amount = equity * account_risk_pct
        price_risk = abs(entry_price - stop_price)
        if price_risk == 0:
            return 0
        shares = int(risk_amount / price_risk)
        return max(shares, 0)


class BacktestEngine:
    """백테스팅 엔진"""

    def __init__(
        self,
        strategy,
        data: dict[str, pd.DataFrame],
        initial_capital: float = 100_000_000,
        commission_rate: float = 0.00015,
        tax_rate: float = 0.0023,
    ):
        self.strategy = strategy
        self.data = data  # {ticker: DataFrame}
        self.portfolio = Portfolio(initial_capital, commission_rate, tax_rate)
        self.dates: list[datetime] = []

    def _get_all_dates(self) -> list:
        """모든 티커의 날짜를 합쳐서 정렬"""
        all_dates = set()
        for df in self.data.values():
            all_dates.update(df.index.tolist())
        return sorted(all_dates)

    def _get_prices_on_date(self, date) -> dict[str, float]:
        """특정 날짜의 종가"""
        prices = {}
        for ticker, df in self.data.items():
            if date in df.index:
                prices[ticker] = df.loc[date, "Close"]
        return prices

    def _get_data_up_to(self, date) -> dict[str, pd.DataFrame]:
        """특정 날짜까지의 데이터"""
        result = {}
        for ticker, df in self.data.items():
            mask = df.index <= date
            if mask.any():
                result[ticker] = df.loc[mask].copy()
        return result

    def run(self) -> dict:
        """백테스트 실행"""
        self.dates = self._get_all_dates()

        if not self.dates:
            raise ValueError("No data available for backtesting")

        logger.info(
            f"백테스트 시작: {self.strategy.name} | "
            f"{self.dates[0]:%Y-%m-%d} ~ {self.dates[-1]:%Y-%m-%d} | "
            f"종목 수: {len(self.data)}"
        )

        # 전략 초기화
        self.strategy.initialize(self.data)

        for i, date in enumerate(self.dates):
            prices = self._get_prices_on_date(date)
            historical = self._get_data_up_to(date)

            if not prices:
                continue

            # 손절 체크
            self._check_stop_losses(date, prices)

            # 전략 시그널 생성
            signals = self.strategy.generate_signals(
                date=date,
                historical_data=historical,
                portfolio=self.portfolio,
                current_prices=prices,
            )

            # 시그널 실행
            self._execute_signals(date, signals, prices)

            # 자산 기록
            self.portfolio.record_equity(date, prices)

        logger.info(
            f"백테스트 완료: 총 거래 {len(self.portfolio.trades)}건, "
            f"청산 {len(self.portfolio.closed_trades)}건"
        )

        return self._compile_results()

    def _check_stop_losses(self, date, prices: dict[str, float]) -> None:
        """손절 체크"""
        tickers_to_sell = []
        for ticker, pos in self.portfolio.positions.items():
            if pos.stop_loss and ticker in prices:
                if prices[ticker] <= pos.stop_loss:
                    tickers_to_sell.append(ticker)

        for ticker in tickers_to_sell:
            self.portfolio.sell(
                ticker=ticker,
                date=date,
                price=prices[ticker],
                strategy=self.portfolio.positions[ticker].strategy,
                reason="손절",
            )

    def _execute_signals(self, date, signals: list[dict], prices: dict) -> None:
        """시그널 실행"""
        for signal in signals:
            ticker = signal["ticker"]
            action = signal["action"]

            if ticker not in prices:
                continue

            price = prices[ticker]

            if action == "BUY":
                shares = signal.get("shares", 0)
                stop_loss = signal.get("stop_loss")
                reason = signal.get("reason", "")

                if shares > 0:
                    self.portfolio.buy(
                        ticker=ticker,
                        date=date,
                        price=price,
                        shares=shares,
                        strategy=self.strategy.name,
                        reason=reason,
                        stop_loss=stop_loss,
                    )
            elif action == "SELL":
                shares = signal.get("shares")
                reason = signal.get("reason", "")

                self.portfolio.sell(
                    ticker=ticker,
                    date=date,
                    price=price,
                    shares=shares,
                    strategy=self.strategy.name,
                    reason=reason,
                )

    def _compile_results(self) -> dict:
        """결과 종합"""
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        if equity_df.empty:
            return {"error": "No equity data"}

        equity_df.set_index("date", inplace=True)
        closed_df = pd.DataFrame(self.portfolio.closed_trades)

        return {
            "strategy": self.strategy.name,
            "equity_curve": equity_df,
            "trades": self.portfolio.trades,
            "closed_trades": closed_df,
            "portfolio": self.portfolio,
            "initial_capital": self.portfolio.initial_capital,
            "final_equity": equity_df["equity"].iloc[-1] if len(equity_df) > 0 else 0,
        }
