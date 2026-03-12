"""
펀더멘털 지표 계산 모듈
- 가치/성장/배당 전략에 필요한 재무 지표
- 실제 데이터 없이도 시뮬레이션 가능한 프록시 포함
"""

import numpy as np
import pandas as pd


class FundamentalIndicators:
    """펀더멘털 지표 계산"""

    # ── 밸류에이션 ──

    @staticmethod
    def per(price: float, eps: float) -> float:
        """주가수익비율 (PER)"""
        if eps <= 0:
            return float("inf")
        return price / eps

    @staticmethod
    def pbr(price: float, bps: float) -> float:
        """주가순자산비율 (PBR)"""
        if bps <= 0:
            return float("inf")
        return price / bps

    @staticmethod
    def psr(market_cap: float, revenue: float) -> float:
        """주가매출비율 (PSR)"""
        if revenue <= 0:
            return float("inf")
        return market_cap / revenue

    @staticmethod
    def ev_ebitda(ev: float, ebitda: float) -> float:
        """EV/EBITDA"""
        if ebitda <= 0:
            return float("inf")
        return ev / ebitda

    @staticmethod
    def peg(per: float, eps_growth: float) -> float:
        """PEG ratio: PER ÷ EPS성장률"""
        if eps_growth <= 0:
            return float("inf")
        return per / eps_growth

    # ── 수익성 ──

    @staticmethod
    def roe(net_income: float, equity: float) -> float:
        """자기자본이익률 (ROE)"""
        if equity <= 0:
            return 0.0
        return net_income / equity * 100

    @staticmethod
    def roic(nopat: float, invested_capital: float) -> float:
        """투하자본수익률 (ROIC)"""
        if invested_capital <= 0:
            return 0.0
        return nopat / invested_capital * 100

    @staticmethod
    def fcf_yield(fcf: float, market_cap: float) -> float:
        """FCF Yield = 잉여현금흐름 / 시가총액"""
        if market_cap <= 0:
            return 0.0
        return fcf / market_cap * 100

    # ── 안전성 ──

    @staticmethod
    def debt_ratio(total_debt: float, equity: float) -> float:
        """부채비율"""
        if equity <= 0:
            return float("inf")
        return total_debt / equity * 100

    @staticmethod
    def interest_coverage(operating_income: float, interest_expense: float) -> float:
        """이자보상배율"""
        if interest_expense <= 0:
            return float("inf")
        return operating_income / interest_expense

    @staticmethod
    def altman_z_score(
        working_capital: float,
        retained_earnings: float,
        ebit: float,
        market_cap: float,
        total_liabilities: float,
        sales: float,
        total_assets: float,
    ) -> float:
        """알트만 Z-Score (부도 예측)"""
        if total_assets == 0:
            return 0.0
        a = working_capital / total_assets
        b = retained_earnings / total_assets
        c = ebit / total_assets
        d = market_cap / max(total_liabilities, 1)
        e = sales / total_assets
        return 1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + 1.0 * e

    # ── 이익의 질 ──

    @staticmethod
    def accrual_ratio(net_income: float, ocf: float, total_assets: float) -> float:
        """발생액 비율: (순이익 - OCF) / 총자산"""
        if total_assets == 0:
            return 0.0
        return (net_income - ocf) / total_assets

    # ── 배당 ──

    @staticmethod
    def dividend_yield(annual_dividend: float, price: float) -> float:
        """배당수익률"""
        if price <= 0:
            return 0.0
        return annual_dividend / price * 100

    @staticmethod
    def payout_ratio(dividend: float, net_income: float) -> float:
        """배당성향 (순이익 대비)"""
        if net_income <= 0:
            return float("inf")
        return dividend / net_income * 100

    @staticmethod
    def fcf_payout_ratio(dividend: float, fcf: float) -> float:
        """FCF 배당성향"""
        if fcf <= 0:
            return float("inf")
        return dividend / fcf * 100

    @staticmethod
    def dividend_coverage(fcf: float, total_dividend: float) -> float:
        """배당 커버리지 = FCF / 총배당금"""
        if total_dividend <= 0:
            return float("inf")
        return fcf / total_dividend

    # ── 복합 모델 ──

    @staticmethod
    def piotroski_f_score(metrics: dict) -> int:
        """
        피오트로스키 F-Score (9점 만점)
        metrics에 필요한 키:
        - roa, prev_roa, ocf, ocf_vs_ni,
        - debt_ratio, prev_debt_ratio,
        - current_ratio, prev_current_ratio,
        - shares_outstanding, prev_shares_outstanding,
        - gross_margin, prev_gross_margin,
        - asset_turnover, prev_asset_turnover
        """
        score = 0

        # 수익성 (4점)
        if metrics.get("roa", 0) > 0:
            score += 1
        if metrics.get("ocf", 0) > 0:
            score += 1
        if metrics.get("roa", 0) > metrics.get("prev_roa", 0):
            score += 1
        if metrics.get("ocf", 0) > metrics.get("roa", 0) * metrics.get("total_assets", 1):
            score += 1

        # 레버리지/유동성 (3점)
        if metrics.get("debt_ratio", 100) < metrics.get("prev_debt_ratio", 100):
            score += 1
        if metrics.get("current_ratio", 0) > metrics.get("prev_current_ratio", 0):
            score += 1
        if metrics.get("shares_outstanding", 1) <= metrics.get("prev_shares_outstanding", 1):
            score += 1

        # 영업 효율성 (2점)
        if metrics.get("gross_margin", 0) > metrics.get("prev_gross_margin", 0):
            score += 1
        if metrics.get("asset_turnover", 0) > metrics.get("prev_asset_turnover", 0):
            score += 1

        return score

    @staticmethod
    def greenblatt_magic_formula_rank(
        roic_values: dict[str, float],
        earnings_yield: dict[str, float],
    ) -> dict[str, int]:
        """
        그린블라트 마법공식 순위
        ROIC 순위 + 이익수익률 순위의 합산 랭킹
        """
        tickers = set(roic_values.keys()) & set(earnings_yield.keys())

        roic_sorted = sorted(tickers, key=lambda t: roic_values.get(t, 0), reverse=True)
        ey_sorted = sorted(tickers, key=lambda t: earnings_yield.get(t, 0), reverse=True)

        roic_rank = {t: i + 1 for i, t in enumerate(roic_sorted)}
        ey_rank = {t: i + 1 for i, t in enumerate(ey_sorted)}

        combined = {t: roic_rank[t] + ey_rank[t] for t in tickers}
        final_sorted = sorted(combined.items(), key=lambda x: x[1])

        return {t: rank + 1 for rank, (t, _) in enumerate(final_sorted)}
