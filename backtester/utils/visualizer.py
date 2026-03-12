"""
시각화 모듈
- 수익 곡선, 드로다운, 거래 분포, 전략 비교 차트
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# 한글 폰트 설정
plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


class Visualizer:
    """백테스트 결과 시각화"""

    COLORS = {
        "포지션 트레이딩": "#2196F3",
        "성장주 트레이딩": "#FF5722",
        "가치 트레이딩": "#4CAF50",
        "스윙 트레이딩": "#9C27B0",
        "모멘텀 트레이딩": "#FF9800",
        "배당 투자": "#607D8B",
        "벤치마크": "#999999",
    }

    @staticmethod
    def plot_equity_curve(
        results: dict,
        benchmark: pd.DataFrame = None,
        save_path: str = None,
    ) -> None:
        """수익 곡선 차트"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[3, 1, 1])
        fig.suptitle(
            f"백테스트 결과: {results['strategy']}",
            fontsize=16,
            fontweight="bold",
        )

        equity = results["equity_curve"]
        initial = results["initial_capital"]
        color = Visualizer.COLORS.get(results["strategy"], "#2196F3")

        # 1. 자산 곡선
        ax1 = axes[0]
        ax1.plot(
            equity.index, equity["equity"],
            color=color, linewidth=1.5, label=results["strategy"],
        )
        ax1.axhline(y=initial, color="gray", linestyle="--", alpha=0.5, label="초기자본")

        if benchmark is not None and not benchmark.empty:
            # 벤치마크를 초기자본 기준으로 정규화
            bench_norm = benchmark["Close"] / benchmark["Close"].iloc[0] * initial
            ax1.plot(
                bench_norm.index, bench_norm,
                color="#999999", linewidth=1, alpha=0.7, label="벤치마크",
            )

        ax1.set_ylabel("자산 (원)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"{x / 1e6:.0f}M" if x >= 1e6 else f"{x:,.0f}")
        )

        # 2. 드로다운
        ax2 = axes[1]
        peak = equity["equity"].cummax()
        drawdown = (equity["equity"] - peak) / peak * 100
        ax2.fill_between(equity.index, drawdown, 0, color="red", alpha=0.3)
        ax2.set_ylabel("드로다운 (%)")
        ax2.grid(True, alpha=0.3)

        # 3. 보유 종목 수
        ax3 = axes[2]
        ax3.fill_between(
            equity.index, equity["num_positions"],
            color=color, alpha=0.3,
        )
        ax3.set_ylabel("보유 종목")
        ax3.set_xlabel("날짜")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"차트 저장: {save_path}")

        plt.close()

    @staticmethod
    def plot_trade_distribution(results: dict, save_path: str = None) -> None:
        """거래 수익률 분포"""
        closed = results.get("closed_trades")
        if closed is None or closed.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"거래 분석: {results['strategy']}",
            fontsize=14,
            fontweight="bold",
        )

        # 1. 수익률 히스토그램
        ax1 = axes[0]
        pnl_pcts = closed["pnl_pct"]
        colors = ["#4CAF50" if x > 0 else "#F44336" for x in pnl_pcts]
        ax1.bar(range(len(pnl_pcts)), pnl_pcts.values, color=colors, alpha=0.7)
        ax1.axhline(y=0, color="black", linewidth=0.5)
        ax1.set_xlabel("거래 번호")
        ax1.set_ylabel("수익률 (%)")
        ax1.set_title("개별 거래 수익률")
        ax1.grid(True, alpha=0.3)

        # 2. 수익률 분포
        ax2 = axes[1]
        ax2.hist(pnl_pcts, bins=30, color="#2196F3", alpha=0.7, edgecolor="white")
        ax2.axvline(
            x=pnl_pcts.mean(), color="red", linestyle="--",
            label=f"평균: {pnl_pcts.mean():.2f}%",
        )
        ax2.axvline(x=0, color="black", linewidth=0.5)
        ax2.set_xlabel("수익률 (%)")
        ax2.set_ylabel("빈도")
        ax2.set_title("수익률 분포")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close()

    @staticmethod
    def plot_strategy_comparison(
        all_results: list[dict],
        save_path: str = None,
    ) -> None:
        """전략 간 비교 차트"""
        if not all_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("전략 비교 분석", fontsize=16, fontweight="bold")

        # 1. 수익 곡선 비교
        ax1 = axes[0, 0]
        for r in all_results:
            eq = r["equity_curve"]
            initial = r["initial_capital"]
            norm = eq["equity"] / initial * 100
            color = Visualizer.COLORS.get(r["strategy"], "#333")
            ax1.plot(eq.index, norm, label=r["strategy"], color=color, linewidth=1.2)

        ax1.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
        ax1.set_ylabel("수익률 (%)")
        ax1.set_title("수익 곡선 비교")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. 드로다운 비교
        ax2 = axes[0, 1]
        for r in all_results:
            eq = r["equity_curve"]["equity"]
            peak = eq.cummax()
            dd = (eq - peak) / peak * 100
            color = Visualizer.COLORS.get(r["strategy"], "#333")
            ax2.plot(dd.index, dd, label=r["strategy"], color=color, alpha=0.7)

        ax2.set_ylabel("드로다운 (%)")
        ax2.set_title("드로다운 비교")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. 승률 & 손익비 비교
        ax3 = axes[1, 0]
        from utils.performance import PerformanceAnalyzer

        strategies = []
        win_rates = []
        risk_rewards = []

        for r in all_results:
            m = PerformanceAnalyzer.analyze(r)
            strategies.append(r["strategy"])
            wr_str = m.get("승률", "0%").replace("%", "")
            rr_str = m.get("손익비", "0")
            try:
                win_rates.append(float(wr_str))
                risk_rewards.append(float(rr_str))
            except ValueError:
                win_rates.append(0)
                risk_rewards.append(0)

        x = np.arange(len(strategies))
        w = 0.35
        bars1 = ax3.bar(x - w / 2, win_rates, w, label="승률 (%)", color="#4CAF50", alpha=0.8)
        ax3_twin = ax3.twinx()
        bars2 = ax3_twin.bar(x + w / 2, risk_rewards, w, label="손익비", color="#FF9800", alpha=0.8)

        ax3.set_xticks(x)
        ax3.set_xticklabels([s[:4] for s in strategies], fontsize=9)
        ax3.set_ylabel("승률 (%)")
        ax3_twin.set_ylabel("손익비")
        ax3.set_title("승률 & 손익비")
        ax3.legend(loc="upper left", fontsize=8)
        ax3_twin.legend(loc="upper right", fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4. 총수익률 바 차트
        ax4 = axes[1, 1]
        returns = []
        colors_list = []
        for r in all_results:
            ret = (r["final_equity"] / r["initial_capital"] - 1) * 100
            returns.append(ret)
            colors_list.append(Visualizer.COLORS.get(r["strategy"], "#333"))

        bars = ax4.barh(
            [s[:6] for s in strategies], returns,
            color=colors_list, alpha=0.8,
        )
        ax4.set_xlabel("총수익률 (%)")
        ax4.set_title("총수익률 비교")
        ax4.grid(True, alpha=0.3, axis="x")

        for bar, ret in zip(bars, returns):
            ax4.text(
                bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{ret:.1f}%", va="center", fontsize=9,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close()
