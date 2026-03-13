"""
포트폴리오 핵심 엔진
- 포지션 사이징: (계좌 × 리스크%) ÷ (진입가 - 손절가)
- 상관관계 분석 + 분산 효과 측정
- 포트폴리오 수준 리스크 지표 (VaR, CVaR, MDD)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum


class Strategy(Enum):
    POSITION = ("포지션", 0.45, 3.0)    # (이름, 목표승률, 목표손익비)
    GROWTH = ("성장주", 0.40, 4.0)
    VALUE = ("가치", 0.60, 2.5)
    SWING = ("스윙", 0.55, 1.75)
    MOMENTUM = ("모멘텀", 0.50, 2.0)
    DIVIDEND = ("배당", 0.70, 1.5)

    def __init__(self, label: str, target_wr: float, target_rr: float):
        self.label = label
        self.target_wr = target_wr
        self.target_rr = target_rr

    @property
    def expectancy(self) -> float:
        """기대값 = (승률 × 평균이익) - (패률 × 평균손실)"""
        return self.target_wr * self.target_rr - (1 - self.target_wr)


@dataclass
class Position:
    """개별 포지션"""
    ticker: str
    strategy: Strategy
    entry_price: float
    stop_loss: float
    shares: int
    current_price: float = 0.0
    target_price: float | None = None

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def total_risk(self) -> float:
        return self.risk_per_share * self.shares

    @property
    def position_value(self) -> float:
        price = self.current_price if self.current_price > 0 else self.entry_price
        return price * self.shares

    @property
    def pnl(self) -> float:
        if self.current_price <= 0:
            return 0
        return (self.current_price - self.entry_price) * self.shares

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def r_multiple(self) -> float:
        if self.risk_per_share <= 0:
            return 0
        return (self.current_price - self.entry_price) / self.risk_per_share


@dataclass
class PortfolioStats:
    """포트폴리오 통계"""
    total_value: float = 0
    cash: float = 0
    invested: float = 0
    total_risk: float = 0
    risk_pct: float = 0
    unrealized_pnl: float = 0
    num_positions: int = 0
    strategy_allocation: dict = field(default_factory=dict)
    var_95: float = 0
    var_99: float = 0
    cvar_95: float = 0
    max_drawdown: float = 0
    sharpe_ratio: float = 0
    sortino_ratio: float = 0
    correlation_avg: float = 0
    diversification_ratio: float = 0


class Portfolio:
    """포트폴리오 관리자"""

    def __init__(self, initial_capital: float = 100_000_000,
                 risk_per_trade_pct: float = 1.0,
                 max_portfolio_risk_pct: float = 6.0,
                 max_positions: int = 10,
                 max_strategy_pct: float = 30.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.max_positions = max_positions
        self.max_strategy_pct = max_strategy_pct
        self.positions: list[Position] = []
        self.closed_trades: list[dict] = []
        self.equity_history: list[float] = [initial_capital]
        self.daily_returns: list[float] = []

    # ──────────────────────────────────────────
    #  포지션 사이징 (문서 공식)
    # ──────────────────────────────────────────
    def calculate_position_size(self, entry_price: float, stop_loss: float,
                                 risk_reduction: float = 1.0) -> dict:
        """
        포지션 사이즈 계산
        공식: 주수 = (계좌 × 리스크%) ÷ (진입가 - 손절가)

        risk_reduction: 연패 규칙에 의한 축소 비율 (0.5 = 50% 축소)
        """
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return {"shares": 0, "error": "손절가가 진입가와 같습니다"}

        account_risk = self.capital * (self.risk_per_trade_pct / 100) * risk_reduction
        shares = int(account_risk / risk_per_share)
        position_value = shares * entry_price
        position_pct = position_value / self.capital * 100 if self.capital > 0 else 0

        return {
            "shares": shares,
            "position_value": position_value,
            "position_pct": position_pct,
            "risk_amount": shares * risk_per_share,
            "risk_pct": (shares * risk_per_share) / self.capital * 100,
            "risk_per_share": risk_per_share,
            "risk_reduction": risk_reduction,
        }

    def can_add_position(self, strategy: Strategy, position_value: float) -> tuple[bool, str]:
        """포지션 추가 가능 여부 검증"""
        if len(self.positions) >= self.max_positions:
            return False, f"최대 포지션 수({self.max_positions}) 초과"

        # 전략별 비중 제한
        strat_value = sum(p.position_value for p in self.positions if p.strategy == strategy)
        strat_pct = (strat_value + position_value) / self.capital * 100
        if strat_pct > self.max_strategy_pct:
            return False, f"{strategy.label} 전략 비중 {strat_pct:.1f}% > 한도 {self.max_strategy_pct}%"

        # 포트폴리오 총 리스크 제한
        total_risk = sum(p.total_risk for p in self.positions)
        portfolio_risk_pct = total_risk / self.capital * 100
        if portfolio_risk_pct > self.max_portfolio_risk_pct:
            return False, f"포트폴리오 리스크 {portfolio_risk_pct:.1f}% > 한도 {self.max_portfolio_risk_pct}%"

        return True, "OK"

    def add_position(self, position: Position) -> bool:
        ok, msg = self.can_add_position(position.strategy, position.position_value)
        if not ok:
            print(f"  ⚠ 포지션 추가 불가: {msg}")
            return False
        self.positions.append(position)
        self.capital -= position.position_value
        return True

    # ──────────────────────────────────────────
    #  리스크 지표 계산
    # ──────────────────────────────────────────
    def compute_stats(self, returns_matrix: pd.DataFrame | None = None) -> PortfolioStats:
        """포트폴리오 통계 산출"""
        stats = PortfolioStats()
        stats.num_positions = len(self.positions)
        stats.invested = sum(p.position_value for p in self.positions)
        stats.cash = self.capital
        stats.total_value = stats.cash + stats.invested
        stats.total_risk = sum(p.total_risk for p in self.positions)
        stats.risk_pct = stats.total_risk / stats.total_value * 100 if stats.total_value > 0 else 0
        stats.unrealized_pnl = sum(p.pnl for p in self.positions)

        # 전략별 배분
        for strat in Strategy:
            val = sum(p.position_value for p in self.positions if p.strategy == strat)
            if val > 0:
                stats.strategy_allocation[strat.label] = val / stats.total_value * 100

        # 수익률 기반 지표 (returns_matrix가 있을 경우)
        if returns_matrix is not None and len(returns_matrix) > 20:
            portfolio_returns = self._calculate_portfolio_returns(returns_matrix)
            if len(portfolio_returns) > 0:
                stats.var_95 = np.percentile(portfolio_returns, 5)
                stats.var_99 = np.percentile(portfolio_returns, 1)
                tail = portfolio_returns[portfolio_returns <= stats.var_95]
                stats.cvar_95 = tail.mean() if len(tail) > 0 else stats.var_95

                # 최대 낙폭
                cum = (1 + portfolio_returns).cumprod()
                peak = cum.expanding().max()
                dd = (cum - peak) / peak
                stats.max_drawdown = dd.min()

                # 샤프/소르티노
                rf = 0.035 / 252  # 일일 무위험수익률 (연 3.5%)
                excess = portfolio_returns - rf
                stats.sharpe_ratio = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
                downside = excess[excess < 0]
                down_std = downside.std() if len(downside) > 0 else 1
                stats.sortino_ratio = excess.mean() / down_std * np.sqrt(252) if down_std > 0 else 0

                # 상관관계
                if returns_matrix.shape[1] > 1:
                    corr = returns_matrix.corr()
                    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
                    stats.correlation_avg = corr.where(mask).mean().mean()

                    # 분산 비율
                    weights = np.ones(returns_matrix.shape[1]) / returns_matrix.shape[1]
                    individual_vol = returns_matrix.std()
                    weighted_vol = (weights * individual_vol).sum()
                    portfolio_vol = portfolio_returns.std()
                    stats.diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 1

        return stats

    def _calculate_portfolio_returns(self, returns_matrix: pd.DataFrame) -> pd.Series:
        """포트폴리오 수익률 계산 (균등 비중)"""
        n = returns_matrix.shape[1]
        if n == 0:
            return pd.Series(dtype=float)
        weights = np.ones(n) / n
        return (returns_matrix * weights).sum(axis=1)

    # ──────────────────────────────────────────
    #  포맷팅
    # ──────────────────────────────────────────
    def format_positions(self) -> str:
        if not self.positions:
            return "  포지션 없음"

        lines = [
            f"\n{'─' * 90}",
            f"  {'종목':10s} {'전략':8s} {'진입가':>10s} {'현재가':>10s} {'손절가':>10s} "
            f"{'수량':>6s} {'평가금':>12s} {'손익':>10s} {'R배수':>6s}",
            f"{'─' * 90}",
        ]
        for p in self.positions:
            lines.append(
                f"  {p.ticker:10s} {p.strategy.label:8s} {p.entry_price:>10,.0f} "
                f"{p.current_price:>10,.0f} {p.stop_loss:>10,.0f} {p.shares:>6d} "
                f"{p.position_value:>12,.0f} {p.pnl:>+10,.0f} {p.r_multiple:>+6.1f}R"
            )
        lines.append(f"{'─' * 90}")
        return "\n".join(lines)

    def format_stats(self, stats: PortfolioStats) -> str:
        lines = [
            f"\n{'═' * 60}",
            f"  📊 포트폴리오 요약",
            f"{'═' * 60}",
            f"  총 자산:     {stats.total_value:>15,.0f}원",
            f"  투자금:      {stats.invested:>15,.0f}원",
            f"  현금:        {stats.cash:>15,.0f}원",
            f"  현금 비중:   {stats.cash / stats.total_value * 100:>14.1f}%",
            f"  포지션 수:   {stats.num_positions:>15d}개",
            f"  미실현 손익: {stats.unrealized_pnl:>+15,.0f}원",
            f"\n{'─' * 60}",
            f"  🛡️ 리스크 지표",
            f"{'─' * 60}",
            f"  총 리스크:   {stats.total_risk:>15,.0f}원 ({stats.risk_pct:.2f}%)",
            f"  VaR (95%):   {stats.var_95:>+15.2%} (일일)",
            f"  VaR (99%):   {stats.var_99:>+15.2%} (일일)",
            f"  CVaR (95%):  {stats.cvar_95:>+15.2%} (일일)",
            f"  최대 낙폭:   {stats.max_drawdown:>+15.2%}",
            f"\n{'─' * 60}",
            f"  📈 성과 지표",
            f"{'─' * 60}",
            f"  샤프 비율:   {stats.sharpe_ratio:>15.2f}",
            f"  소르티노:    {stats.sortino_ratio:>15.2f}",
            f"  평균 상관:   {stats.correlation_avg:>15.2f}",
            f"  분산 비율:   {stats.diversification_ratio:>15.2f}",
        ]

        if stats.strategy_allocation:
            lines.extend([
                f"\n{'─' * 60}",
                f"  🎯 전략별 비중",
                f"{'─' * 60}",
            ])
            for strat, pct in sorted(stats.strategy_allocation.items(), key=lambda x: -x[1]):
                bar = "█" * int(pct / 2) + "░" * (25 - int(pct / 2))
                lines.append(f"  {strat:8s} {bar} {pct:5.1f}%")

        return "\n".join(lines)
