#!/usr/bin/env python3
"""
주식 투자 기법 백테스팅 시뮬레이터
====================================
6가지 투자 전략(포지션/성장주/가치/스윙/모멘텀/배당)을
과거 데이터로 검증하고 성과를 비교 분석합니다.

사용법:
    python main.py                    # config.py 설정으로 전체 실행
    python main.py --strategy swing   # 특정 전략만 실행
    python main.py --market kr        # 한국 주식으로 실행
"""

import argparse
import logging
import os
import sys
import time

import pandas as pd
from tabulate import tabulate

from config import (
    DATA_CONFIG,
    OUTPUT_CONFIG,
    PORTFOLIO_CONFIG,
    RUN_STRATEGIES,
    STRATEGY_PARAMS,
)
from engine import BacktestEngine
from strategies import (
    PositionTradingStrategy,
    GrowthTradingStrategy,
    ValueTradingStrategy,
    SwingTradingStrategy,
    MomentumTradingStrategy,
    DividendInvestingStrategy,
)
from utils.data_loader import DataLoader
from utils.performance import PerformanceAnalyzer
from utils.visualizer import Visualizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 전략 매핑
STRATEGY_MAP = {
    "position": ("포지션 트레이딩", PositionTradingStrategy),
    "growth": ("성장주 트레이딩", GrowthTradingStrategy),
    "value": ("가치 트레이딩", ValueTradingStrategy),
    "swing": ("스윙 트레이딩", SwingTradingStrategy),
    "momentum": ("모멘텀 트레이딩", MomentumTradingStrategy),
    "dividend": ("배당 투자", DividendInvestingStrategy),
}


def parse_args():
    parser = argparse.ArgumentParser(description="주식 투자 기법 백테스팅 시뮬레이터")
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        choices=list(STRATEGY_MAP.keys()) + ["all"],
        default="all",
        help="실행할 전략 (기본: all)",
    )
    parser.add_argument(
        "--market", "-m",
        type=str,
        choices=["kr", "us"],
        help="시장 선택 (kr/us)",
    )
    parser.add_argument(
        "--start", type=str, help="시작일 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, help="종료일 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capital", type=float, help="초기자본",
    )
    parser.add_argument(
        "--tickers", "-t", nargs="+", help="종목 코드 리스트",
    )
    parser.add_argument(
        "--no-chart", action="store_true", help="차트 저장 안 함",
    )
    return parser.parse_args()


def load_data(args) -> tuple[dict, pd.DataFrame]:
    """데이터 로드"""
    config = DATA_CONFIG.copy()

    if args.market:
        config["market"] = args.market
    if args.start:
        config["start_date"] = args.start
    if args.end:
        config["end_date"] = args.end
    if args.tickers:
        config["tickers"] = args.tickers

    # 종목 데이터
    if config["tickers"]:
        data = DataLoader.download(
            config["tickers"],
            start=config["start_date"],
            end=config["end_date"],
        )
    elif config["market"] == "kr":
        data = DataLoader.get_korean_stocks(
            n=config["num_stocks"],
            start=config["start_date"],
            end=config["end_date"],
        )
    else:
        data = DataLoader.get_us_stocks(
            n=config["num_stocks"],
            start=config["start_date"],
            end=config["end_date"],
        )

    # 벤치마크
    benchmark_ticker = config["benchmark"]
    if config["market"] == "kr" and not args.market:
        benchmark_ticker = "^KS11"

    benchmark = DataLoader.get_benchmark(
        benchmark_ticker,
        start=config["start_date"],
        end=config["end_date"],
    )

    logger.info(f"로드 완료: {len(data)}개 종목, {config['start_date']} ~ {config['end_date']}")
    return data, benchmark


def run_backtest(
    strategy_key: str,
    data: dict[str, pd.DataFrame],
) -> dict:
    """단일 전략 백테스트 실행"""
    name, cls = STRATEGY_MAP[strategy_key]
    params = STRATEGY_PARAMS.get(strategy_key, {})
    strategy = cls(params)

    engine = BacktestEngine(
        strategy=strategy,
        data=data,
        initial_capital=PORTFOLIO_CONFIG["initial_capital"],
        commission_rate=PORTFOLIO_CONFIG["commission_rate"],
        tax_rate=PORTFOLIO_CONFIG["tax_rate"],
    )

    start_time = time.time()
    results = engine.run()
    elapsed = time.time() - start_time

    logger.info(f"  {name} 완료 ({elapsed:.1f}초)")
    return results


def print_results(results: dict, show_trades: bool = True) -> None:
    """결과 출력"""
    metrics = PerformanceAnalyzer.analyze(results)
    strategy = results.get("strategy", "")

    print(f"\n{'=' * 60}")
    print(f"  {strategy} 백테스트 결과")
    print(f"{'=' * 60}")

    for key, value in metrics.items():
        print(f"  {key:20s} : {value}")

    # 개별 거래 내역
    if show_trades:
        closed = results.get("closed_trades")
        if closed is not None and not closed.empty:
            n = OUTPUT_CONFIG.get("max_trades_display", 20)
            display = closed.head(n).copy()
            display["entry_date"] = display["entry_date"].dt.strftime("%Y-%m-%d")
            display["exit_date"] = display["exit_date"].dt.strftime("%Y-%m-%d")
            display["pnl"] = display["pnl"].apply(lambda x: f"{x:,.0f}")
            display["pnl_pct"] = display["pnl_pct"].apply(lambda x: f"{x:.2f}%")

            print(f"\n  최근 {min(n, len(closed))}건 거래:")
            print(
                tabulate(
                    display[["ticker", "entry_date", "exit_date", "pnl_pct", "reason"]],
                    headers=["종목", "진입일", "청산일", "수익률", "사유"],
                    tablefmt="simple",
                    showindex=False,
                )
            )


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  주식 투자 기법 백테스팅 시뮬레이터")
    print("  Stock Investment Strategy Backtester")
    print("=" * 60)

    # 1. 데이터 로드
    logger.info("데이터 다운로드 중...")
    data, benchmark = load_data(args)

    if not data:
        logger.error("데이터를 로드할 수 없습니다.")
        sys.exit(1)

    # 2. 실행할 전략 결정
    if args.strategy == "all":
        strategies_to_run = [
            k for k, v in RUN_STRATEGIES.items() if v
        ]
    else:
        strategies_to_run = [args.strategy]

    # 3. 백테스트 실행
    all_results = []
    for key in strategies_to_run:
        name, _ = STRATEGY_MAP[key]
        logger.info(f"백테스트 실행: {name}")
        try:
            results = run_backtest(key, data)
            all_results.append(results)
            print_results(results, show_trades=OUTPUT_CONFIG.get("print_trades", True))
        except Exception as e:
            logger.error(f"  {name} 실패: {e}")

    if not all_results:
        logger.error("실행된 전략이 없습니다.")
        sys.exit(1)

    # 4. 전략 비교
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("  전략 비교 요약")
        print(f"{'=' * 60}")

        comparison = PerformanceAnalyzer.compare_strategies(all_results)
        key_cols = ["전략", "총수익률", "CAGR", "MDD", "샤프비율", "승률", "손익비"]
        available = [c for c in key_cols if c in comparison.columns]
        print(tabulate(comparison[available], headers="keys", tablefmt="grid", showindex=False))

    # 5. 차트 저장
    if not args.no_chart and OUTPUT_CONFIG.get("save_charts", True):
        output_dir = OUTPUT_CONFIG.get("output_dir", "results")
        os.makedirs(output_dir, exist_ok=True)

        for r in all_results:
            safe_name = r["strategy"].replace(" ", "_")
            Visualizer.plot_equity_curve(
                r, benchmark=benchmark,
                save_path=os.path.join(output_dir, f"{safe_name}_equity.png"),
            )
            Visualizer.plot_trade_distribution(
                r,
                save_path=os.path.join(output_dir, f"{safe_name}_trades.png"),
            )

        if len(all_results) > 1:
            Visualizer.plot_strategy_comparison(
                all_results,
                save_path=os.path.join(output_dir, "strategy_comparison.png"),
            )

        logger.info(f"차트 저장 완료: {output_dir}/")

    # 6. CSV 저장
    if OUTPUT_CONFIG.get("save_csv", True):
        output_dir = OUTPUT_CONFIG.get("output_dir", "results")
        os.makedirs(output_dir, exist_ok=True)

        for r in all_results:
            safe_name = r["strategy"].replace(" ", "_")

            # 자산 곡선
            r["equity_curve"].to_csv(
                os.path.join(output_dir, f"{safe_name}_equity.csv")
            )

            # 거래 내역
            closed = r.get("closed_trades")
            if closed is not None and not closed.empty:
                closed.to_csv(
                    os.path.join(output_dir, f"{safe_name}_trades.csv"),
                    index=False,
                )

        logger.info(f"CSV 저장 완료: {output_dir}/")

    print(f"\n{'=' * 60}")
    print("  백테스트 완료!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
