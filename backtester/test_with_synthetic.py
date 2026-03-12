#!/usr/bin/env python3
"""
합성 데이터를 이용한 백테스팅 검증 스크립트
네트워크 없이도 전략 로직과 엔진을 테스트합니다.
"""

import os
import sys
import numpy as np
import pandas as pd

from engine import BacktestEngine
from strategies import (
    PositionTradingStrategy,
    GrowthTradingStrategy,
    ValueTradingStrategy,
    SwingTradingStrategy,
    MomentumTradingStrategy,
    DividendInvestingStrategy,
)
from utils.performance import PerformanceAnalyzer
from utils.visualizer import Visualizer
from tabulate import tabulate


def generate_synthetic_stock(
    ticker: str,
    days: int = 1000,
    start_price: float = 50000,
    trend: float = 0.0003,
    volatility: float = 0.02,
    seed: int = None,
) -> pd.DataFrame:
    """합성 주가 데이터 생성 (GBM 모델)"""
    if seed is not None:
        np.random.seed(seed)

    dates = pd.bdate_range(start="2021-01-04", periods=days)
    returns = np.random.normal(trend, volatility, days)
    prices = start_price * np.cumprod(1 + returns)

    # OHLCV 생성
    high = prices * (1 + np.abs(np.random.normal(0, 0.008, days)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.008, days)))
    open_prices = prices * (1 + np.random.normal(0, 0.003, days))
    volume = np.random.randint(100000, 5000000, days).astype(float)

    # 가끔 거래량 폭증 (이벤트)
    spikes = np.random.choice(days, size=days // 50, replace=False)
    volume[spikes] *= np.random.uniform(3, 8, len(spikes))

    df = pd.DataFrame(
        {
            "Open": open_prices,
            "High": np.maximum(high, np.maximum(open_prices, prices)),
            "Low": np.minimum(low, np.minimum(open_prices, prices)),
            "Close": prices,
            "Volume": volume,
        },
        index=dates,
    )
    return df


def main():
    print("\n" + "=" * 60)
    print("  합성 데이터 백테스팅 검증")
    print("=" * 60)

    # ── 1. 합성 데이터 생성 ──
    print("\n[1] 합성 주가 데이터 생성 중...")

    stocks = {
        "STOCK_A": generate_synthetic_stock("A", trend=0.0005, volatility=0.018, seed=42),    # 강한 상승
        "STOCK_B": generate_synthetic_stock("B", trend=0.0002, volatility=0.015, seed=123),   # 완만한 상승
        "STOCK_C": generate_synthetic_stock("C", trend=-0.0001, volatility=0.022, seed=456),  # 약한 하락
        "STOCK_D": generate_synthetic_stock("D", trend=0.0003, volatility=0.025, seed=789),   # 높은 변동성
        "STOCK_E": generate_synthetic_stock("E", trend=0.0001, volatility=0.010, seed=101),   # 안정적 (배당형)
        "STOCK_F": generate_synthetic_stock("F", trend=0.0004, volatility=0.020, seed=202),   # 모멘텀형
        "STOCK_G": generate_synthetic_stock("G", trend=0.00015, volatility=0.012, seed=303),  # 저변동
        "STOCK_H": generate_synthetic_stock("H", trend=0.0006, volatility=0.030, seed=404),   # 고성장 고변동
    }

    for name, df in stocks.items():
        ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        print(f"  {name}: {len(df)}일, {df['Close'].iloc[0]:,.0f} → {df['Close'].iloc[-1]:,.0f} ({ret:+.1f}%)")

    # ── 2. 전략별 백테스트 ──
    strategies = [
        ("position", PositionTradingStrategy),
        ("growth", GrowthTradingStrategy),
        ("value", ValueTradingStrategy),
        ("swing", SwingTradingStrategy),
        ("momentum", MomentumTradingStrategy),
        ("dividend", DividendInvestingStrategy),
    ]

    all_results = []
    initial_capital = 100_000_000

    for key, cls in strategies:
        strategy = cls()
        engine = BacktestEngine(
            strategy=strategy,
            data=stocks,
            initial_capital=initial_capital,
            commission_rate=0.00015,
            tax_rate=0.0023,
        )

        print(f"\n[백테스트] {strategy.name} 실행 중...")
        try:
            results = engine.run()
            all_results.append(results)

            # 성과 출력
            metrics = PerformanceAnalyzer.analyze(results)
            print(f"  {'총수익률':12s}: {metrics.get('총수익률', 'N/A')}")
            print(f"  {'MDD':12s}: {metrics.get('MDD', 'N/A')}")
            print(f"  {'승률':12s}: {metrics.get('승률', 'N/A')}")
            print(f"  {'손익비':12s}: {metrics.get('손익비', 'N/A')}")
            print(f"  {'총 거래 수':12s}: {metrics.get('총 거래 수', 0)}")

        except Exception as e:
            print(f"  ❌ 오류: {e}")
            import traceback
            traceback.print_exc()

    # ── 3. 전략 비교 ──
    if len(all_results) > 1:
        print(f"\n{'=' * 70}")
        print("  전략 비교 요약")
        print(f"{'=' * 70}")

        comparison = PerformanceAnalyzer.compare_strategies(all_results)
        key_cols = ["전략", "총수익률", "CAGR", "MDD", "샤프비율", "승률", "손익비", "총 거래 수"]
        available = [c for c in key_cols if c in comparison.columns]
        print(tabulate(comparison[available], headers="keys", tablefmt="grid", showindex=False))

    # ── 4. 차트 저장 ──
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    for r in all_results:
        safe_name = r["strategy"].replace(" ", "_")
        try:
            Visualizer.plot_equity_curve(
                r, save_path=os.path.join(output_dir, f"{safe_name}_equity.png"),
            )
            Visualizer.plot_trade_distribution(
                r, save_path=os.path.join(output_dir, f"{safe_name}_trades.png"),
            )
        except Exception as e:
            print(f"  차트 오류 ({r['strategy']}): {e}")

    if len(all_results) > 1:
        try:
            Visualizer.plot_strategy_comparison(
                all_results,
                save_path=os.path.join(output_dir, "strategy_comparison.png"),
            )
        except Exception as e:
            print(f"  비교 차트 오류: {e}")

    print(f"\n  차트 저장: {output_dir}/")
    print(f"\n{'=' * 60}")
    print("  검증 완료!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
