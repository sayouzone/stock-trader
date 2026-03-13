#!/usr/bin/env python3
"""
시장 레짐 감지기 CLI
- 시장 데이터를 분석하여 현재 레짐(강세/약세/횡보) 판별
- 레짐에 최적인 투자 전략 추천
- 레짐 이력 시각화
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import compute_all
from regime_detector import RegimeDetector, Regime
from strategy_advisor import StrategyAdvisor
from visualizer import plot_regime_timeline, plot_strategy_heatmap, plot_regime_transitions


# ──────────────────────────────────────────────
#  합성 데이터 생성 (다양한 시장 시나리오)
# ──────────────────────────────────────────────
def generate_market_scenario(scenario: str = "mixed", days: int = 750) -> pd.DataFrame:
    """
    시장 시나리오별 합성 데이터 생성

    시나리오:
    - "bull": 지속적 강세장
    - "bear": 지속적 약세장
    - "mixed": 강세→횡보→약세→회복 (가장 현실적)
    - "volatile": 고변동성 장세
    """
    rng = np.random.default_rng(42)

    if scenario == "bull":
        # 꾸준한 상승 (연 15% + 노이즈)
        mu_seq = np.full(days, 0.0006)
        sigma_seq = np.full(days, 0.015)

    elif scenario == "bear":
        # 꾸준한 하락 (연 -20% + 높은 변동성)
        mu_seq = np.full(days, -0.0008)
        sigma_seq = np.full(days, 0.025)

    elif scenario == "volatile":
        # 고변동성 횡보
        mu_seq = np.full(days, 0.0001)
        sigma_seq = np.full(days, 0.035)

    else:  # "mixed" (기본)
        # 4개 구간으로 분할: 강세→횡보→약세→회복
        seg = days // 4
        mu_seq = np.concatenate([
            np.full(seg, 0.0008),       # Phase 1: 강세 (연 ~20%)
            np.full(seg, 0.0001),       # Phase 2: 횡보
            np.full(seg, -0.0010),      # Phase 3: 약세 (연 ~-25%)
            np.full(days - 3 * seg, 0.0006),  # Phase 4: 회복
        ])
        sigma_seq = np.concatenate([
            np.full(seg, 0.012),        # 저변동성 강세
            np.full(seg, 0.018),        # 중간 변동성
            np.full(seg, 0.028),        # 고변동성 약세
            np.full(days - 3 * seg, 0.020),  # 변동성 감소
        ])

    # GBM 기반 가격 생성
    prices = [100.0]
    for i in range(days - 1):
        ret = mu_seq[i] + sigma_seq[i] * rng.standard_normal()
        prices.append(prices[-1] * np.exp(ret))

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)
    close = np.array(prices)
    high = close * (1 + rng.uniform(0.002, 0.025, days))
    low = close * (1 - rng.uniform(0.002, 0.025, days))
    opn = close * (1 + rng.uniform(-0.01, 0.01, days))
    volume = rng.integers(1_000_000, 10_000_000, days).astype(float)

    # 약세장 구간에서 거래량 급증
    if scenario == "mixed":
        bear_start = seg * 2
        bear_end = seg * 3
        volume[bear_start:bear_end] *= rng.uniform(1.5, 3.0, bear_end - bear_start)

    return pd.DataFrame({
        "Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume
    }, index=dates)


def try_yfinance(ticker: str, period: str = "3y") -> pd.DataFrame | None:
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False)
        if df is not None and len(df) > 200:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────
#  메인 분석 로직
# ──────────────────────────────────────────────
def analyze(ticker: str, df: pd.DataFrame, output_dir: str, show_detail: bool = True):
    """단일 종목/시장 분석 실행"""
    detector = RegimeDetector()
    advisor = StrategyAdvisor()

    # 지표 계산
    df = compute_all(df)

    # 현재 레짐 판별
    current = detector.detect_single(df)

    print(f"\n{'═' * 60}")
    print(f"  📊 {ticker} 시장 레짐 분석")
    print(f"{'═' * 60}")

    # 전략 추천
    recommendation = advisor.format_recommendation(current)
    print(recommendation)

    if show_detail:
        # 레짐 이력 산출
        print(f"\n  ⏳ 레짐 이력 분석 중...")
        history = detector.detect_history(df)
        print(f"  ✅ {len(history)}일 분석 완료")

        # 레짐 전환 탐지
        transitions = detector.find_transitions(history)
        print(f"\n{'─' * 60}")
        print(f"  🔄 레짐 전환 이벤트: {len(transitions)}회")
        print(f"{'─' * 60}")

        for t in transitions[-10:]:  # 최근 10건
            print(f"  {t['date'].strftime('%Y-%m-%d')}  "
                  f"{t['from'].label} → {t['to'].label}  "
                  f"(점수: {t['score_before']:+.2f} → {t['score_after']:+.2f})  "
                  f"가격: {t['price']:,.0f}")

        # 레짐 통계
        regime_stats = {}
        for r in history:
            regime_stats[r.regime] = regime_stats.get(r.regime, 0) + 1
        total = len(history)

        print(f"\n{'─' * 60}")
        print(f"  📈 레짐 분포 통계")
        print(f"{'─' * 60}")
        for regime in Regime:
            count = regime_stats.get(regime, 0)
            pct = count / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"  {regime.label:12s} {bar} {pct:5.1f}% ({count}일)")

        # 시각화
        print(f"\n  📊 차트 생성 중...")
        os.makedirs(output_dir, exist_ok=True)

        plot_regime_timeline(
            df, history,
            title=f"{ticker} 시장 레짐 타임라인",
            save_path=os.path.join(output_dir, f"{ticker}_regime_timeline.png")
        )

        plot_strategy_heatmap(
            history,
            save_path=os.path.join(output_dir, f"{ticker}_strategy_heatmap.png")
        )

        if transitions:
            plot_regime_transitions(
                transitions,
                save_path=os.path.join(output_dir, f"{ticker}_transitions.png")
            )

        print(f"\n{'═' * 60}")
        print(f"  ✅ 분석 완료! 차트 저장 위치: {output_dir}/")
        print(f"{'═' * 60}\n")

    return current


def main():
    parser = argparse.ArgumentParser(
        description="📊 시장 레짐 감지기 — 강세/약세/횡보 판별 + 최적 전략 추천",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-t", "--ticker", default="KOSPI",
                        help="분석할 종목/지수 티커 (기본: KOSPI)")
    parser.add_argument("-s", "--scenario", default="mixed",
                        choices=["bull", "bear", "mixed", "volatile"],
                        help="합성 데이터 시나리오 (기본: mixed)")
    parser.add_argument("--days", type=int, default=750,
                        help="합성 데이터 일수 (기본: 750)")
    parser.add_argument("--synthetic", action="store_true", default=False,
                        help="합성 데이터 강제 사용")
    parser.add_argument("--brief", action="store_true", default=False,
                        help="간략 출력 (이력/차트 생략)")
    parser.add_argument("--output", default=None,
                        help="차트 출력 디렉토리")
    parser.add_argument("--demo", action="store_true", default=False,
                        help="데모 모드: 4가지 시나리오 전체 분석")

    args = parser.parse_args()

    output_dir = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    if args.demo:
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║" + "  🎬 데모 모드: 4가지 시장 시나리오 분석".center(52) + "║")
        print("╚" + "═" * 58 + "╝")

        for scenario in ["bull", "bear", "mixed", "volatile"]:
            label = {"bull": "강세장", "bear": "약세장", "mixed": "혼합장", "volatile": "고변동성"}[scenario]
            ticker = f"DEMO_{label}"
            print(f"\n{'▓' * 60}")
            print(f"  시나리오: {label} ({scenario})")
            print(f"{'▓' * 60}")
            df = generate_market_scenario(scenario, days=args.days)
            analyze(ticker, df, output_dir, show_detail=True)
        return

    # 단일 분석
    if args.synthetic:
        df = generate_market_scenario(args.scenario, days=args.days)
        source = f"합성 ({args.scenario})"
    else:
        df = try_yfinance(args.ticker)
        if df is None:
            print(f"  ⚠ {args.ticker}: yfinance 실패 → 합성 데이터({args.scenario})로 대체")
            df = generate_market_scenario(args.scenario, days=args.days)
            source = f"합성 대체 ({args.scenario})"
        else:
            source = "실시간"

    print(f"\n  데이터: {source} | {len(df)}일")
    analyze(args.ticker, df, output_dir, show_detail=not args.brief)


if __name__ == "__main__":
    main()
