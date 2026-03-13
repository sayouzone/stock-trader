"""
포트폴리오 최적화 엔진
- 평균-분산 최적화 (마코위츠)
- 리스크 패리티
- 켈리 기준 (Kelly Criterion)
- 효율적 프론티어 산출
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from portfolio import Strategy


@dataclass
class OptimizationResult:
    """최적화 결과"""
    method: str
    weights: dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    description: str


class PortfolioOptimizer:
    """포트폴리오 비중 최적화기"""

    def __init__(self, returns: pd.DataFrame, rf_rate: float = 0.035):
        """
        returns: 자산별 일일 수익률 DataFrame (컬럼 = 자산명)
        rf_rate: 연간 무위험수익률
        """
        self.returns = returns
        self.rf_daily = rf_rate / 252
        self.rf_annual = rf_rate
        self.n_assets = returns.shape[1]
        self.asset_names = list(returns.columns)

        # 연율화된 수익률/리스크
        self.mean_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252
        self.std_returns = returns.std() * np.sqrt(252)

    # ──────────────────────────────────────────
    #  1. 최소 분산 포트폴리오
    # ──────────────────────────────────────────
    def min_variance(self) -> OptimizationResult:
        """최소 분산 포트폴리오 (수치 해석)"""
        n = self.n_assets
        cov_inv = np.linalg.pinv(self.cov_matrix.values)
        ones = np.ones(n)
        weights = cov_inv @ ones / (ones @ cov_inv @ ones)
        weights = np.maximum(weights, 0)
        weights /= weights.sum()

        ret, risk = self._portfolio_performance(weights)
        sharpe = (ret - self.rf_annual) / risk if risk > 0 else 0

        return OptimizationResult(
            method="최소 분산",
            weights=dict(zip(self.asset_names, weights)),
            expected_return=ret,
            expected_risk=risk,
            sharpe_ratio=sharpe,
            description="리스크를 최소화하는 보수적 배분"
        )

    # ──────────────────────────────────────────
    #  2. 최대 샤프 포트폴리오 (접선 포트폴리오)
    # ──────────────────────────────────────────
    def max_sharpe(self, n_portfolios: int = 20000) -> OptimizationResult:
        """몬테카를로 시뮬레이션으로 최대 샤프 비율 포트폴리오 탐색"""
        best_sharpe = -999
        best_weights = None
        rng = np.random.default_rng(42)

        for _ in range(n_portfolios):
            w = rng.random(self.n_assets)
            w /= w.sum()
            ret, risk = self._portfolio_performance(w)
            sharpe = (ret - self.rf_annual) / risk if risk > 0 else 0
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = w

        ret, risk = self._portfolio_performance(best_weights)
        return OptimizationResult(
            method="최대 샤프 (접선)",
            weights=dict(zip(self.asset_names, best_weights)),
            expected_return=ret,
            expected_risk=risk,
            sharpe_ratio=best_sharpe,
            description="위험 대비 수익이 최대인 배분"
        )

    # ──────────────────────────────────────────
    #  3. 리스크 패리티
    # ──────────────────────────────────────────
    def risk_parity(self, max_iter: int = 500) -> OptimizationResult:
        """각 자산의 리스크 기여도를 균등하게 배분"""
        n = self.n_assets
        cov = self.cov_matrix.values
        w = np.ones(n) / n

        for _ in range(max_iter):
            port_var = w @ cov @ w
            port_vol = np.sqrt(port_var)
            mrc = cov @ w / port_vol  # 한계 리스크 기여
            rc = w * mrc              # 리스크 기여
            target_rc = port_vol / n
            w_new = w * target_rc / (rc + 1e-12)
            w_new = np.maximum(w_new, 0)
            w_new /= w_new.sum()
            if np.max(np.abs(w_new - w)) < 1e-8:
                break
            w = w_new

        ret, risk = self._portfolio_performance(w)
        sharpe = (ret - self.rf_annual) / risk if risk > 0 else 0

        return OptimizationResult(
            method="리스크 패리티",
            weights=dict(zip(self.asset_names, w)),
            expected_return=ret,
            expected_risk=risk,
            sharpe_ratio=sharpe,
            description="모든 자산의 리스크 기여도를 균등하게 배분"
        )

    # ──────────────────────────────────────────
    #  4. 켈리 기준 (전략별 비중)
    # ──────────────────────────────────────────
    def kelly_criterion(self) -> OptimizationResult:
        """
        켈리 기준: f* = (bp - q) / b
        b = 평균 이익 / 평균 손실 (손익비)
        p = 승률, q = 패률
        """
        weights = {}
        for name in self.asset_names:
            # 수익률 기반 승률/손익비 추정
            rets = self.returns[name].dropna()
            wins = rets[rets > 0]
            losses = rets[rets < 0]

            if len(wins) == 0 or len(losses) == 0:
                weights[name] = 1.0 / self.n_assets
                continue

            p = len(wins) / len(rets)
            q = 1 - p
            b = wins.mean() / abs(losses.mean())

            kelly_f = (b * p - q) / b
            # 반켈리 (보수적): 실무에서는 풀 켈리의 50%
            half_kelly = max(0, kelly_f * 0.5)
            weights[name] = half_kelly

        # 정규화
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            weights = {k: 1.0 / self.n_assets for k in self.asset_names}

        w_arr = np.array([weights[n] for n in self.asset_names])
        ret, risk = self._portfolio_performance(w_arr)
        sharpe = (ret - self.rf_annual) / risk if risk > 0 else 0

        return OptimizationResult(
            method="켈리 기준 (반켈리)",
            weights=weights,
            expected_return=ret,
            expected_risk=risk,
            sharpe_ratio=sharpe,
            description="기대값 기반 최적 베팅 비율 (보수적 반켈리 적용)"
        )

    # ──────────────────────────────────────────
    #  5. 효율적 프론티어
    # ──────────────────────────────────────────
    def efficient_frontier(self, n_points: int = 50, n_portfolios: int = 15000) -> pd.DataFrame:
        """효율적 프론티어 포인트 산출"""
        rng = np.random.default_rng(42)
        results = []

        for _ in range(n_portfolios):
            w = rng.random(self.n_assets)
            w /= w.sum()
            ret, risk = self._portfolio_performance(w)
            sharpe = (ret - self.rf_annual) / risk if risk > 0 else 0
            results.append({"return": ret, "risk": risk, "sharpe": sharpe,
                            **{f"w_{name}": w[i] for i, name in enumerate(self.asset_names)}})

        df = pd.DataFrame(results)

        # 프론티어 추출 (리스크 구간별 최대 수익률)
        risk_min, risk_max = df["risk"].min(), df["risk"].max()
        risk_bins = np.linspace(risk_min, risk_max, n_points)
        frontier = []
        for i in range(len(risk_bins) - 1):
            mask = (df["risk"] >= risk_bins[i]) & (df["risk"] < risk_bins[i + 1])
            subset = df[mask]
            if len(subset) > 0:
                best = subset.loc[subset["return"].idxmax()]
                frontier.append(best)

        return pd.DataFrame(frontier).reset_index(drop=True)

    # ──────────────────────────────────────────
    #  유틸리티
    # ──────────────────────────────────────────
    def _portfolio_performance(self, weights: np.ndarray) -> tuple[float, float]:
        """(연율 수익률, 연율 표준편차)"""
        ret = weights @ self.mean_returns.values
        var = weights @ self.cov_matrix.values @ weights
        return ret, np.sqrt(var)

    def compare_all(self) -> list[OptimizationResult]:
        """4가지 방법 비교"""
        return [
            self.min_variance(),
            self.max_sharpe(),
            self.risk_parity(),
            self.kelly_criterion(),
        ]

    def format_comparison(self, results: list[OptimizationResult]) -> str:
        lines = [
            f"\n{'═' * 70}",
            f"  📊 포트폴리오 최적화 비교",
            f"{'═' * 70}",
        ]

        for r in sorted(results, key=lambda x: -x.sharpe_ratio):
            lines.extend([
                f"\n  ▸ {r.method}  (샤프 {r.sharpe_ratio:.2f})",
                f"    {r.description}",
                f"    기대 수익률: {r.expected_return:.1%}  |  리스크: {r.expected_risk:.1%}",
                f"    비중:",
            ])
            for asset, w in sorted(r.weights.items(), key=lambda x: -x[1]):
                if w >= 0.01:
                    bar = "█" * int(w * 40) + "░" * (40 - int(w * 40))
                    lines.append(f"      {asset:12s} {bar} {w:5.1%}")

        return "\n".join(lines)
