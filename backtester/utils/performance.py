"""
성과 분석 모듈
- 수익률, MDD, 샤프비율, 승률, 손익비 등 핵심 지표 계산
- 문서의 기법별 목표와 비교 분석
"""

import numpy as np
import pandas as pd


class PerformanceAnalyzer:
    """백테스트 성과 분석"""

    # 문서의 기법별 목표치
    STRATEGY_TARGETS = {
        "포지션 트레이딩": {"win_rate": (40, 50), "risk_reward": 3.0},
        "성장주 트레이딩": {"win_rate": (35, 45), "risk_reward": 3.0},
        "가치 트레이딩": {"win_rate": (55, 65), "risk_reward": 2.0},
        "스윙 트레이딩": {"win_rate": (50, 60), "risk_reward": 2.0},
        "모멘텀 트레이딩": {"win_rate": (45, 55), "risk_reward": 2.0},
        "배당 투자": {"win_rate": (65, 75), "risk_reward": 1.5},
    }

    @staticmethod
    def analyze(results: dict) -> dict:
        """종합 성과 분석"""
        equity_curve = results.get("equity_curve")
        closed_trades = results.get("closed_trades")
        initial = results.get("initial_capital", 100_000_000)
        strategy_name = results.get("strategy", "")

        if equity_curve is None or equity_curve.empty:
            return {"error": "No data"}

        metrics = {}

        # ── 수익률 ──
        final_equity = equity_curve["equity"].iloc[-1]
        total_return = (final_equity / initial - 1) * 100

        trading_days = len(equity_curve)
        years = trading_days / 252
        if years > 0:
            cagr = ((final_equity / initial) ** (1 / years) - 1) * 100
        else:
            cagr = 0

        metrics["초기자본"] = f"{initial:,.0f}"
        metrics["최종자본"] = f"{final_equity:,.0f}"
        metrics["총수익률"] = f"{total_return:.2f}%"
        metrics["CAGR"] = f"{cagr:.2f}%"

        # ── 리스크 지표 ──
        equity_series = equity_curve["equity"]
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak * 100
        mdd = drawdown.min()
        metrics["MDD"] = f"{mdd:.2f}%"

        # 변동성
        daily_returns = equity_series.pct_change().dropna()
        if len(daily_returns) > 0:
            annual_vol = daily_returns.std() * np.sqrt(252) * 100
            metrics["연간변동성"] = f"{annual_vol:.2f}%"

            # 샤프 비율 (무위험이자율 3% 가정)
            rf = 0.03
            excess = daily_returns.mean() * 252 - rf
            if daily_returns.std() > 0:
                sharpe = excess / (daily_returns.std() * np.sqrt(252))
            else:
                sharpe = 0
            metrics["샤프비율"] = f"{sharpe:.2f}"

            # 소르티노 비율
            downside = daily_returns[daily_returns < 0]
            if len(downside) > 0 and downside.std() > 0:
                sortino = excess / (downside.std() * np.sqrt(252))
                metrics["소르티노비율"] = f"{sortino:.2f}"

            # Calmar 비율
            if mdd != 0:
                calmar = cagr / abs(mdd)
                metrics["칼마비율"] = f"{calmar:.2f}"

        # ── 거래 분석 ──
        if closed_trades is not None and not closed_trades.empty:
            ct = closed_trades
            total_trades = len(ct)
            winners = ct[ct["pnl"] > 0]
            losers = ct[ct["pnl"] <= 0]

            win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
            metrics["총 거래 수"] = total_trades
            metrics["승률"] = f"{win_rate:.1f}%"

            # 평균 수익/손실
            avg_win = winners["pnl_pct"].mean() if len(winners) > 0 else 0
            avg_loss = abs(losers["pnl_pct"].mean()) if len(losers) > 0 else 1
            metrics["평균 수익"] = f"{avg_win:.2f}%"
            metrics["평균 손실"] = f"{-abs(avg_loss) if avg_loss else 0:.2f}%"

            # 손익비
            if avg_loss > 0:
                risk_reward = avg_win / avg_loss
            else:
                risk_reward = float("inf")
            metrics["손익비"] = f"{risk_reward:.2f}"

            # 기대값
            expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
            metrics["기대값 (%)"] = f"{expectancy:.2f}%"

            # 최대 연승/연패
            pnl_signs = (ct["pnl"] > 0).astype(int)
            streaks = pnl_signs.groupby((pnl_signs != pnl_signs.shift()).cumsum())
            if len(streaks) > 0:
                max_win_streak = max(
                    (len(g) for _, g in streaks if g.iloc[0] == 1), default=0
                )
                max_loss_streak = max(
                    (len(g) for _, g in streaks if g.iloc[0] == 0), default=0
                )
                metrics["최대 연승"] = max_win_streak
                metrics["최대 연패"] = max_loss_streak

            # 평균 보유 기간
            if "holding_days" in ct.columns:
                metrics["평균 보유일"] = f"{ct['holding_days'].mean():.1f}일"

            # 최고/최저 수익 거래
            metrics["최대 수익 거래"] = f"{ct['pnl_pct'].max():.2f}%"
            metrics["최대 손실 거래"] = f"{ct['pnl_pct'].min():.2f}%"

            # ── 목표 대비 달성도 ──
            target = PerformanceAnalyzer.STRATEGY_TARGETS.get(strategy_name)
            if target:
                wr_low, wr_high = target["win_rate"]
                wr_status = "✅" if wr_low <= win_rate <= wr_high else "❌"
                rr_status = "✅" if risk_reward >= target["risk_reward"] else "❌"

                metrics["[목표] 승률"] = (
                    f"{wr_low}~{wr_high}% {wr_status} (실제: {win_rate:.1f}%)"
                )
                metrics["[목표] 손익비"] = (
                    f"{target['risk_reward']}:1 이상 {rr_status} (실제: {risk_reward:.2f})"
                )

        return metrics

    @staticmethod
    def compare_strategies(all_results: list[dict]) -> pd.DataFrame:
        """여러 전략 성과 비교 테이블"""
        rows = []
        for results in all_results:
            metrics = PerformanceAnalyzer.analyze(results)
            row = {"전략": results.get("strategy", "")}
            row.update(metrics)
            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def monthly_returns(equity_curve: pd.DataFrame) -> pd.DataFrame:
        """월별 수익률 테이블"""
        if equity_curve.empty:
            return pd.DataFrame()

        monthly = equity_curve["equity"].resample("ME").last()
        monthly_ret = monthly.pct_change() * 100

        # 연도-월 피벗
        table = pd.DataFrame(
            {
                "year": monthly_ret.index.year,
                "month": monthly_ret.index.month,
                "return": monthly_ret.values,
            }
        )

        pivot = table.pivot(index="year", columns="month", values="return")
        pivot.columns = [
            f"{m}월" for m in pivot.columns
        ]

        # 연간 수익률 추가
        yearly = equity_curve["equity"].resample("YE").last().pct_change() * 100
        pivot["연간"] = yearly.values[: len(pivot)]

        return pivot.round(2)
