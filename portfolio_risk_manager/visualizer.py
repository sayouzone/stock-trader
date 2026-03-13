"""
포트폴리오 리스크 시각화 모듈
- 효율적 프론티어
- 리스크 기여도 분해
- 상관관계 히트맵
- 포지션 사이징 차트
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# 한글 폰트 설정
for font in ["NanumGothic", "Malgun Gothic", "AppleGothic", "DejaVu Sans"]:
    try:
        matplotlib.rcParams["font.family"] = font
        matplotlib.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue


def plot_efficient_frontier(frontier_df: pd.DataFrame,
                             optimization_results: list,
                             save_path: str | None = None):
    """효율적 프론티어 + 최적 포트폴리오 표시"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # 프론티어 곡선
    ax.scatter(frontier_df["risk"] * 100, frontier_df["return"] * 100,
               c=frontier_df["sharpe"], cmap="RdYlGn", s=5, alpha=0.6, zorder=1)

    # 최적 포트폴리오 표시
    markers = ["★", "◆", "▲", "●"]
    colors = ["#FF4444", "#4444FF", "#44AA44", "#FF8800"]
    for i, r in enumerate(optimization_results):
        ax.scatter(r.expected_risk * 100, r.expected_return * 100,
                   s=200, c=colors[i], marker="o", zorder=5,
                   edgecolors="black", linewidth=1.5, label=f"{r.method} (S={r.sharpe_ratio:.2f})")

    # CML (Capital Market Line)
    best = max(optimization_results, key=lambda x: x.sharpe_ratio)
    x_cml = np.linspace(0, best.expected_risk * 200, 50)
    y_cml = 3.5 + best.sharpe_ratio * x_cml  # rf + sharpe * sigma
    ax.plot(x_cml, y_cml, "k--", alpha=0.4, linewidth=1, label="CML")

    ax.set_xlabel("Risk (Ann. Std Dev %)", fontsize=12)
    ax.set_ylabel("Return (Ann. %)", fontsize=12)
    ax.set_title("Efficient Frontier", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.colorbar(ax.collections[0], ax=ax, label="Sharpe Ratio", shrink=0.8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  > {save_path}")
    plt.close(fig)


def plot_correlation_heatmap(returns: pd.DataFrame, save_path: str | None = None):
    """상관관계 히트맵"""
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = LinearSegmentedColormap.from_list("rb", ["#4444FF", "#FFFFFF", "#FF4444"])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    n = len(corr)
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold" if abs(val) > 0.5 else "normal")

    ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Correlation", shrink=0.8)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  > {save_path}")
    plt.close(fig)


def plot_risk_contribution(weights: dict, returns: pd.DataFrame, save_path: str | None = None):
    """리스크 기여도 분해 차트"""
    cov = returns.cov() * 252
    asset_names = list(weights.keys())
    w = np.array([weights[n] for n in asset_names])

    port_var = w @ cov.values @ w
    port_vol = np.sqrt(port_var)
    mrc = cov.values @ w / port_vol
    rc = w * mrc
    rc_pct = rc / rc.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 파이: 비중
    ax1 = axes[0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(asset_names)))
    nonzero_mask = w > 0.01
    ax1.pie(w[nonzero_mask] * 100,
            labels=[asset_names[i] for i in range(len(asset_names)) if nonzero_mask[i]],
            colors=colors[nonzero_mask],
            autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    ax1.set_title("Weight Allocation", fontsize=12, fontweight="bold")

    # 바: 리스크 기여도
    ax2 = axes[1]
    y_pos = range(len(asset_names))
    bars = ax2.barh(y_pos, rc_pct, color=colors, edgecolor="gray")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(asset_names, fontsize=10)
    ax2.set_xlabel("Risk Contribution (%)", fontsize=11)
    ax2.set_title("Risk Contribution", fontsize=12, fontweight="bold")

    for bar, pct in zip(bars, rc_pct):
        if pct > 2:
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{pct:.1f}%", va="center", fontsize=9)

    ax2.axvline(100 / len(asset_names), color="red", linestyle="--", alpha=0.5,
                label=f"Equal ({100 / len(asset_names):.1f}%)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  > {save_path}")
    plt.close(fig)


def plot_position_sizing(capital: float, risk_pct: float = 1.0, save_path: str | None = None):
    """포지션 사이징 시뮬레이션 차트"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 리스크% 별 포지션 크기
    ax1 = axes[0]
    risk_levels = [0.5, 1.0, 1.5, 2.0]
    stop_distances = np.arange(1, 11, 0.5)  # 진입가-손절가 거리 (%)

    for r in risk_levels:
        risk_amount = capital * r / 100
        position_pcts = [risk_amount / (capital * d / 100) * 100 for d in stop_distances]
        ax1.plot(stop_distances, position_pcts, label=f"Risk {r}%", linewidth=2)

    ax1.set_xlabel("Stop Distance (%)", fontsize=11)
    ax1.set_ylabel("Position Size (% of Capital)", fontsize=11)
    ax1.set_title("Position Sizing: Risk % vs Stop Distance", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # 2. 연패에 따른 리스크 축소
    ax2 = axes[1]
    scenarios = ["Normal", "2 Loss", "3 Loss\n(50%)", "4 Loss\n(50%)", "5 Loss\n(STOP)"]
    factors = [1.0, 1.0, 0.5, 0.5, 0.0]
    risk_amounts = [capital * risk_pct / 100 * f for f in factors]
    colors = ["#4CAF50", "#FFC107", "#FF9800", "#FF5722", "#D32F2F"]

    bars = ax2.bar(scenarios, risk_amounts, color=colors, edgecolor="gray")
    ax2.set_ylabel("Risk Amount (Won)", fontsize=11)
    ax2.set_title("Consecutive Loss Rule", fontsize=12, fontweight="bold")

    for bar, amt, f in zip(bars, risk_amounts, factors):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + capital * 0.001,
                 f"{amt:,.0f}\n({f:.0%})", ha="center", fontsize=9)

    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  > {save_path}")
    plt.close(fig)


def plot_risk_dashboard(stats, save_path: str | None = None):
    """리스크 대시보드 종합 차트"""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Portfolio Risk Dashboard", fontsize=16, fontweight="bold", y=0.98)

    # 2x2 레이아웃
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # 1. 전략별 배분 (파이)
    ax1 = fig.add_subplot(gs[0, 0])
    if stats.strategy_allocation:
        labels = list(stats.strategy_allocation.keys())
        sizes = list(stats.strategy_allocation.values())
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 9})
    ax1.set_title("Strategy Allocation", fontsize=12, fontweight="bold")

    # 2. 리스크 게이지
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ["Risk %", "VaR 95%", "MDD", "Cash %"]
    values = [
        stats.risk_pct,
        abs(stats.var_95) * 100,
        abs(stats.max_drawdown) * 100,
        stats.cash / stats.total_value * 100 if stats.total_value > 0 else 0,
    ]
    limits = [6.0, 3.0, 20.0, 100.0]
    colors_bar = []
    for v, lim in zip(values, limits):
        ratio = v / lim if lim > 0 else 0
        if ratio < 0.5:
            colors_bar.append("#4CAF50")
        elif ratio < 0.8:
            colors_bar.append("#FFC107")
        else:
            colors_bar.append("#FF5722")

    y_pos = range(len(metrics))
    ax2.barh(y_pos, values, color=colors_bar, edgecolor="gray")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(metrics, fontsize=10)
    for i, (v, lim) in enumerate(zip(values, limits)):
        ax2.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=9)
        ax2.axvline(lim, color="red", linestyle="--", alpha=0.3)
    ax2.set_title("Risk Metrics", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")

    # 3. 성과 지표
    ax3 = fig.add_subplot(gs[1, 0])
    perf_metrics = ["Sharpe", "Sortino", "Diversification"]
    perf_values = [stats.sharpe_ratio, stats.sortino_ratio, stats.diversification_ratio]
    bar_colors = ["#2196F3" if v > 1 else "#FF9800" if v > 0 else "#F44336" for v in perf_values]
    bars = ax3.bar(perf_metrics, perf_values, color=bar_colors, edgecolor="gray")
    ax3.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="Target = 1.0")
    for bar, v in zip(bars, perf_values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{v:.2f}", ha="center", fontsize=10)
    ax3.set_title("Performance Ratios", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. 요약 텍스트
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    summary = (
        f"Total Value: {stats.total_value:,.0f}\n"
        f"Invested: {stats.invested:,.0f}\n"
        f"Cash: {stats.cash:,.0f}\n"
        f"Positions: {stats.num_positions}\n"
        f"Unrealized PnL: {stats.unrealized_pnl:+,.0f}\n"
        f"─────────────────\n"
        f"Total Risk: {stats.risk_pct:.2f}%\n"
        f"VaR 95%: {stats.var_95:+.2%}\n"
        f"CVaR 95%: {stats.cvar_95:+.2%}\n"
        f"Max Drawdown: {stats.max_drawdown:+.2%}\n"
        f"─────────────────\n"
        f"Sharpe: {stats.sharpe_ratio:.2f}\n"
        f"Sortino: {stats.sortino_ratio:.2f}\n"
        f"Avg Correlation: {stats.correlation_avg:.2f}\n"
        f"Diversification: {stats.diversification_ratio:.2f}"
    )
    ax4.text(0.1, 0.95, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax4.set_title("Summary", fontsize=12, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  > {save_path}")
    plt.close(fig)
