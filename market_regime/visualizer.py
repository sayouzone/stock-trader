"""
시장 레짐 시각화 모듈
- 가격 차트 + 레짐 배경색
- 레짐 점수 타임라인
- 전략 적합도 히트맵
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from regime_detector import Regime, RegimeResult


# 한글 폰트 설정
def setup_korean_font():
    for font in ["NanumGothic", "Malgun Gothic", "AppleGothic", "DejaVu Sans"]:
        try:
            matplotlib.rcParams["font.family"] = font
            matplotlib.rcParams["axes.unicode_minus"] = False
            fig = plt.figure()
            fig.text(0.5, 0.5, "테스트")
            plt.close(fig)
            return
        except Exception:
            continue
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False

setup_korean_font()


REGIME_COLORS = {
    Regime.STRONG_BULL: "#00C853",
    Regime.BULL: "#69F0AE",
    Regime.SIDEWAYS: "#FFD54F",
    Regime.BEAR: "#FF8A80",
    Regime.STRONG_BEAR: "#D32F2F",
}


def plot_regime_timeline(df: pd.DataFrame, history: list[RegimeResult],
                         title: str = "시장 레짐 타임라인",
                         save_path: str | None = None):
    """가격 차트 + 레짐 배경색 시각화"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    dates = [r.date for r in history]
    prices = [r.close_price for r in history]
    scores = [r.composite_score for r in history]

    # ── 1. 가격 차트 + 레짐 배경 ──
    ax1 = axes[0]
    ax1.plot(dates, prices, color="black", linewidth=1.2, label="종가", zorder=3)

    # 200MA
    sma200_vals = df.loc[df.index.isin(dates), "SMA_200"] if "SMA_200" in df.columns else None
    if sma200_vals is not None and not sma200_vals.isna().all():
        ax1.plot(sma200_vals.index, sma200_vals.values, color="blue",
                 linewidth=0.8, alpha=0.7, linestyle="--", label="200MA", zorder=2)

    # 레짐 배경색
    for i in range(1, len(history)):
        color = REGIME_COLORS[history[i].regime]
        ax1.axvspan(dates[i - 1], dates[i], color=color, alpha=0.15, zorder=0)

    ax1.set_ylabel("가격")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── 2. 레짐 점수 ──
    ax2 = axes[1]
    colors = ["#D32F2F" if s < -0.4 else "#00C853" if s > 0.4 else "#FFD54F" for s in scores]
    ax2.bar(dates, scores, color=colors, width=1.5, alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axhline(1.2, color="green", linewidth=0.5, linestyle=":", alpha=0.5)
    ax2.axhline(0.4, color="lightgreen", linewidth=0.5, linestyle=":", alpha=0.5)
    ax2.axhline(-0.4, color="lightsalmon", linewidth=0.5, linestyle=":", alpha=0.5)
    ax2.axhline(-1.2, color="red", linewidth=0.5, linestyle=":", alpha=0.5)
    ax2.set_ylabel("레짐 점수")
    ax2.set_ylim(-2.2, 2.2)
    ax2.grid(True, alpha=0.3)

    # ── 3. 레짐 구간 바 차트 ──
    ax3 = axes[2]
    regime_nums = []
    for r in history:
        regime_map = {Regime.STRONG_BULL: 2, Regime.BULL: 1, Regime.SIDEWAYS: 0,
                      Regime.BEAR: -1, Regime.STRONG_BEAR: -2}
        regime_nums.append(regime_map[r.regime])

    regime_colors_list = [REGIME_COLORS[r.regime] for r in history]
    ax3.bar(dates, [1] * len(dates), color=regime_colors_list, width=1.5)
    ax3.set_yticks([])
    ax3.set_ylabel("레짐")

    # 범례
    legend_patches = [mpatches.Patch(color=REGIME_COLORS[r], label=r.label) for r in Regime]
    ax3.legend(handles=legend_patches, loc="upper center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.3))

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  💾 저장: {save_path}")
    plt.close(fig)
    return fig


def plot_strategy_heatmap(history: list[RegimeResult],
                          save_path: str | None = None):
    """레짐 비율 + 전략 적합도 히트맵"""
    from strategy_advisor import REGIME_STRATEGY_MAP

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                              gridspec_kw={"width_ratios": [1, 2.5]})

    # ── 1. 레짐 비율 파이차트 ──
    ax1 = axes[0]
    regime_counts = {}
    for r in history:
        regime_counts[r.regime] = regime_counts.get(r.regime, 0) + 1
    total = len(history)

    labels = [f"{r.label}\n({c / total:.0%})" for r, c in regime_counts.items()]
    sizes = list(regime_counts.values())
    colors = [REGIME_COLORS[r] for r in regime_counts.keys()]

    ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
            startangle=90, textprops={"fontsize": 9})
    ax1.set_title("레짐 분포", fontsize=12, fontweight="bold")

    # ── 2. 전략 적합도 히트맵 ──
    ax2 = axes[1]
    strategies = ["포지션 트레이딩", "성장주 트레이딩", "가치 트레이딩",
                  "스윙 트레이딩", "모멘텀 트레이딩", "배당 투자"]
    regimes = list(Regime)

    matrix = np.zeros((len(strategies), len(regimes)))
    for j, regime in enumerate(regimes):
        fits = REGIME_STRATEGY_MAP.get(regime, [])
        for fit in fits:
            if fit.name in strategies:
                i = strategies.index(fit.name)
                matrix[i, j] = fit.score

    cmap = LinearSegmentedColormap.from_list("rg", ["#FF6B6B", "#FFE066", "#69DB7C"])
    im = ax2.imshow(matrix, cmap=cmap, aspect="auto", vmin=1, vmax=5)

    ax2.set_xticks(range(len(regimes)))
    ax2.set_xticklabels([r.label for r in regimes], fontsize=9)
    ax2.set_yticks(range(len(strategies)))
    ax2.set_yticklabels(strategies, fontsize=10)

    # 셀에 점수 표시
    for i in range(len(strategies)):
        for j in range(len(regimes)):
            val = int(matrix[i, j])
            stars = "★" * val + "☆" * (5 - val)
            ax2.text(j, i, f"{val}\n{stars}", ha="center", va="center", fontsize=7,
                     color="black" if val >= 3 else "gray")

    ax2.set_title("레짐별 전략 적합도", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax2, label="적합도 (1~5)", shrink=0.8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  💾 저장: {save_path}")
    plt.close(fig)
    return fig


def plot_regime_transitions(transitions: list[dict], save_path: str | None = None):
    """레짐 전환 이벤트 차트"""
    if not transitions:
        print("  전환 이벤트 없음")
        return None

    fig, ax = plt.subplots(figsize=(14, 5))

    dates = [t["date"] for t in transitions]
    scores = [t["score_after"] for t in transitions]
    colors = [REGIME_COLORS[t["to"]] for t in transitions]

    ax.scatter(dates, scores, c=colors, s=100, zorder=5, edgecolors="black", linewidth=0.5)

    for t in transitions:
        label = f"{t['from'].label.split()[-1]}→{t['to'].label.split()[-1]}"
        ax.annotate(label, (t["date"], t["score_after"]),
                    textcoords="offset points", xytext=(0, 15),
                    ha="center", fontsize=7, rotation=45)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("전환 후 레짐 점수")
    ax.set_title("레짐 전환 이벤트", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  💾 저장: {save_path}")
    plt.close(fig)
    return fig
