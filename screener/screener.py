#!/usr/bin/env python3
"""
실시간 종목 스크리너 CLI
- 6가지 투자 기법에 기반한 종목 스코어링
- yfinance로 데이터 다운로드 또는 합성 데이터 테스트
"""
import argparse
import sys
import os

import numpy as np
import pandas as pd

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import compute_all
from scoring import ScreenResult
from screeners import ALL_SCREENERS


# ──────────────────────────────────────────────
#  합성 데이터 생성 (yfinance 대체)
# ──────────────────────────────────────────────
def generate_synthetic(ticker: str, days: int = 500, seed: int | None = None) -> pd.DataFrame:
    """GBM 기반 합성 주가 데이터 생성"""
    rng = np.random.default_rng(seed)
    mu, sigma = 0.0003, 0.02
    prices = [100.0]
    for _ in range(days - 1):
        ret = mu + sigma * rng.standard_normal()
        prices.append(prices[-1] * np.exp(ret))

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)
    close = np.array(prices)
    high = close * (1 + rng.uniform(0.001, 0.03, days))
    low = close * (1 - rng.uniform(0.001, 0.03, days))
    opn = close * (1 + rng.uniform(-0.01, 0.01, days))
    volume = rng.integers(500_000, 5_000_000, days).astype(float)
    # 가끔 거래량 급등
    spikes = rng.choice(days, size=days // 20, replace=False)
    volume[spikes] *= rng.uniform(2, 5, len(spikes))

    return pd.DataFrame({
        "Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume
    }, index=dates)


def try_yfinance(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """yfinance 시도, 실패 시 None 반환"""
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False)
        if df is not None and len(df) > 50:
            # MultiIndex 처리
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────
#  메인 스크리닝 로직
# ──────────────────────────────────────────────
def run_screen(tickers: list[str], strategies: list[str], use_synthetic: bool = False) -> list[ScreenResult]:
    """종목 리스트를 스크리닝"""
    results = []

    for ticker in tickers:
        # 데이터 로딩
        if use_synthetic:
            seed = hash(ticker) % (2**31)
            df = generate_synthetic(ticker, days=500, seed=seed)
            source = "합성"
        else:
            df = try_yfinance(ticker)
            if df is None:
                print(f"  ⚠ {ticker}: yfinance 실패 → 합성 데이터로 대체")
                seed = hash(ticker) % (2**31)
                df = generate_synthetic(ticker, days=500, seed=seed)
                source = "합성(대체)"
            else:
                source = "실시간"

        # 기술적 지표 계산
        df = compute_all(df)

        # 선택된 전략으로 스크리닝
        for strat_key in strategies:
            if strat_key not in ALL_SCREENERS:
                print(f"  ⚠ 알 수 없는 전략: {strat_key}")
                continue
            label, screener_cls = ALL_SCREENERS[strat_key]
            result = screener_cls.screen(ticker, df)
            results.append(result)

    return results


# ──────────────────────────────────────────────
#  출력 포맷팅
# ──────────────────────────────────────────────
def print_summary_table(results: list[ScreenResult]):
    """요약 테이블 출력"""
    if not results:
        print("  결과 없음")
        return

    rows = [r.summary_row() for r in results]
    # 점수 순 정렬
    paired = sorted(zip(results, rows), key=lambda x: x[0].total_score, reverse=True)

    header = ["종목", "전략", "점수", "통과", "등급", "현재가", "손절가", "시그널"]
    widths = [max(len(str(r[h])) for r in [p[1] for p in paired] + [{h: h for h in header}]) for h in header]
    widths = [max(w, len(h)) for w, h in zip(widths, header)]

    sep = "─" * (sum(widths) + 3 * (len(header) - 1) + 4)
    print(f"\n{sep}")
    print("  " + " │ ".join(f"{h:^{w}}" for h, w in zip(header, widths)))
    print(f"{sep}")
    for res, row in paired:
        line = " │ ".join(f"{str(row[h]):^{w}}" for h, w in zip(header, widths))
        print(f"  {line}")
    print(f"{sep}")


def print_details(results: list[ScreenResult], top_n: int = 0):
    """상세 체크리스트 출력"""
    sorted_results = sorted(results, key=lambda x: x.total_score, reverse=True)
    if top_n > 0:
        sorted_results = sorted_results[:top_n]

    for r in sorted_results:
        print(r.detail_str())


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────
def main():
    strat_names = {k: v[0] for k, v in ALL_SCREENERS.items()}
    strat_help = ", ".join(f"{k}({v})" for k, v in strat_names.items())

    parser = argparse.ArgumentParser(
        description="📊 주식 종목 스크리너 — 6가지 투자 기법 기반 종합 스코어링",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-t", "--tickers", nargs="+", default=["삼성전자", "SK하이닉스", "NAVER", "카카오", "LG에너지솔루션"],
        help="분석할 종목 티커 목록 (기본: 한국 대표 5종목)"
    )
    parser.add_argument(
        "-s", "--strategies", nargs="+", default=["all"],
        help=f"적용할 전략 (all=전체)\n사용 가능: {strat_help}"
    )
    parser.add_argument(
        "--synthetic", action="store_true", default=False,
        help="합성 데이터 사용 (yfinance 대신 GBM 생성)"
    )
    parser.add_argument(
        "--detail", action="store_true", default=False,
        help="상세 체크리스트 출력"
    )
    parser.add_argument(
        "--top", type=int, default=0,
        help="상세 출력 시 상위 N개만 표시 (0=전체)"
    )
    parser.add_argument(
        "--demo", action="store_true", default=False,
        help="데모 모드: 10개 합성 종목 × 전체 전략"
    )

    args = parser.parse_args()

    # 데모 모드
    if args.demo:
        args.tickers = [
            "DEMO_삼성전자", "DEMO_SK하이닉스", "DEMO_NAVER", "DEMO_카카오",
            "DEMO_LG에너지", "DEMO_셀트리온", "DEMO_현대차", "DEMO_POSCO",
            "DEMO_삼성SDI", "DEMO_KB금융"
        ]
        args.strategies = ["all"]
        args.synthetic = True
        args.detail = True
        args.top = 5

    # 전략 선택
    if "all" in args.strategies:
        strategies = list(ALL_SCREENERS.keys())
    else:
        strategies = args.strategies

    print("\n" + "=" * 60)
    print("  📊 종목 스크리너 실행")
    print("=" * 60)
    print(f"  종목: {', '.join(args.tickers)}")
    print(f"  전략: {', '.join(strat_names.get(s, s) for s in strategies)}")
    print(f"  데이터: {'합성 (GBM)' if args.synthetic else 'yfinance (실패 시 합성 대체)'}")
    print("=" * 60)

    results = run_screen(args.tickers, strategies, use_synthetic=args.synthetic)

    if not results:
        print("\n  ❌ 스크리닝 결과가 없습니다.")
        return

    # 요약 테이블
    print_summary_table(results)

    # 상세 체크리스트
    if args.detail:
        print(f"\n{'=' * 55}")
        print(f"  📋 상세 체크리스트 {'(상위 ' + str(args.top) + '개)' if args.top > 0 else ''}")
        print(f"{'=' * 55}")
        print_details(results, top_n=args.top)

    # 매수 추천 요약
    buy_signals = [r for r in results if r.entry_signals and r.grade.value in ["★★★★★", "★★★★☆"]]
    if buy_signals:
        print(f"\n{'=' * 55}")
        print("  🔥 매수 시그널 발생 종목")
        print(f"{'=' * 55}")
        for r in sorted(buy_signals, key=lambda x: x.total_score, reverse=True):
            print(f"\n  {r.ticker} ({r.strategy}) — {r.grade.value}")
            for sig in r.entry_signals:
                print(f"    → {sig}")
            if r.stop_loss:
                print(f"    🛑 손절가: {r.stop_loss:,.0f}")
            if r.target_price:
                print(f"    🎯 목표가: {r.target_price:,.0f}")

    # 통계
    print(f"\n{'─' * 55}")
    avg_score = sum(r.score_pct for r in results) / len(results)
    print(f"  전체 평균 점수: {avg_score:.0f}점 | 총 {len(results)}건 분석")
    print(f"  매수 시그널: {len(buy_signals)}건")
    print(f"{'─' * 55}\n")


if __name__ == "__main__":
    main()
