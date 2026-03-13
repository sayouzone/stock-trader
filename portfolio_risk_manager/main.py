#!/usr/bin/env python3
"""
포트폴리오 리스크 매니저 CLI
- 포지션 사이징 계산기
- 포트폴리오 최적화 (4가지 방법)
- 리스크 모니터링 (연패/월간 규칙)
- 종합 대시보드 + 시각화
"""
import argparse
import sys
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from portfolio import Portfolio, Position, Strategy, PortfolioStats
from risk_monitor import RiskMonitor, TradeRecord
from optimizer import PortfolioOptimizer
from visualizer import (
    plot_efficient_frontier, plot_correlation_heatmap,
    plot_risk_contribution, plot_position_sizing, plot_risk_dashboard,
)


# ──────────────────────────────────────────────
#  합성 데이터 생성
# ──────────────────────────────────────────────
def generate_synthetic_returns(n_assets: int = 8, n_days: int = 500) -> pd.DataFrame:
    """다양한 상관구조를 가진 합성 수익률 데이터"""
    rng = np.random.default_rng(42)
    names = ["삼성전자", "SK하이닉스", "NAVER", "카카오",
             "LG에너지", "셀트리온", "현대차", "KB금융",
             "삼성SDI", "POSCO"][:n_assets]

    # 상관관계 행렬 생성 (블록 구조)
    corr = np.eye(n_assets)
    # IT 그룹 (높은 상관)
    for i in range(min(4, n_assets)):
        for j in range(i + 1, min(4, n_assets)):
            corr[i, j] = corr[j, i] = rng.uniform(0.4, 0.7)
    # 나머지 (낮은 상관)
    for i in range(4, n_assets):
        for j in range(i + 1, n_assets):
            corr[i, j] = corr[j, i] = rng.uniform(-0.1, 0.3)

    # 촐레스키 분해로 상관된 수익률 생성
    L = np.linalg.cholesky(corr + np.eye(n_assets) * 0.01)
    raw = rng.standard_normal((n_days, n_assets))
    correlated = raw @ L.T

    # 각 자산에 다른 기대수익률/변동성 적용
    mus = rng.uniform(0.0001, 0.0008, n_assets)
    sigmas = rng.uniform(0.015, 0.035, n_assets)
    returns = mus + correlated * sigmas

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    return pd.DataFrame(returns, index=dates, columns=names)


def generate_demo_trades(monitor: RiskMonitor, n_trades: int = 30):
    """데모 매매 기록 생성"""
    rng = np.random.default_rng(123)
    strategies = list(Strategy)
    base_date = date.today() - timedelta(days=60)

    for i in range(n_trades):
        strat = rng.choice(strategies)
        is_win = rng.random() < strat.target_wr
        if is_win:
            r_mult = rng.uniform(0.5, strat.target_rr * 1.5)
            pnl = r_mult * 1_000_000 * rng.uniform(0.5, 1.5)
        else:
            r_mult = -rng.uniform(0.3, 1.2)
            pnl = r_mult * 1_000_000 * rng.uniform(0.5, 1.5)

        trade_date = base_date + timedelta(days=int(i * 2))
        monitor.add_trade(TradeRecord(
            date=trade_date, strategy=strat,
            ticker=f"종목{i + 1:02d}", pnl=pnl,
            r_multiple=r_mult, is_win=is_win,
        ))


def generate_demo_positions(portfolio: Portfolio, returns: pd.DataFrame):
    """데모 포지션 생성"""
    rng = np.random.default_rng(55)
    strat_map = {
        "삼성전자": Strategy.POSITION,
        "SK하이닉스": Strategy.GROWTH,
        "NAVER": Strategy.MOMENTUM,
        "카카오": Strategy.SWING,
        "LG에너지": Strategy.VALUE,
        "셀트리온": Strategy.DIVIDEND,
        "현대차": Strategy.POSITION,
        "KB금융": Strategy.DIVIDEND,
    }

    for name in returns.columns[:6]:
        # 합성 가격 생성
        cum = (1 + returns[name]).cumprod()
        price = cum.iloc[-1] * 50000

        entry = price * rng.uniform(0.90, 1.05)
        stop = entry * (1 - rng.uniform(0.03, 0.08))
        strategy = strat_map.get(name, Strategy.POSITION)

        sizing = portfolio.calculate_position_size(entry, stop)
        shares = min(sizing["shares"], 500)

        pos = Position(
            ticker=name, strategy=strategy,
            entry_price=round(entry),
            stop_loss=round(stop),
            shares=max(shares, 10),
            current_price=round(price),
        )
        portfolio.add_position(pos)


# ──────────────────────────────────────────────
#  메인 실행
# ──────────────────────────────────────────────
def run_demo(output_dir: str):
    """종합 데모 실행"""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "  📊 포트폴리오 리스크 매니저 — 종합 데모".center(48) + "║")
    print("╚" + "═" * 58 + "╝")

    # 1. 합성 데이터 생성
    print("\n  ⏳ 합성 수익률 데이터 생성 (8종목 × 500일)...")
    returns = generate_synthetic_returns(n_assets=8, n_days=500)
    print(f"  ✅ 생성 완료: {returns.shape}")

    # 2. 포트폴리오 설정
    print("\n  ⏳ 포트폴리오 구성 중...")
    portfolio = Portfolio(
        initial_capital=100_000_000,
        risk_per_trade_pct=1.0,
        max_portfolio_risk_pct=6.0,
        max_positions=10,
        max_strategy_pct=30.0,
    )
    generate_demo_positions(portfolio, returns)
    stats = portfolio.compute_stats(returns)

    # 포지션 현황
    print(portfolio.format_positions())
    print(portfolio.format_stats(stats))

    # 3. 포지션 사이징 계산기
    print(f"\n{'═' * 60}")
    print(f"  📐 포지션 사이징 계산기")
    print(f"{'═' * 60}")
    examples = [
        ("삼성전자", 72000, 67000, Strategy.POSITION),
        ("SK하이닉스", 195000, 180000, Strategy.GROWTH),
        ("NAVER", 210000, 195000, Strategy.SWING),
    ]
    for ticker, entry, stop, strat in examples:
        sizing = portfolio.calculate_position_size(entry, stop)
        print(f"\n  {ticker} ({strat.label})")
        print(f"    진입가: {entry:,}원 | 손절가: {stop:,}원 | 리스크/주: {sizing['risk_per_share']:,.0f}원")
        print(f"    매수 수량: {sizing['shares']:,}주")
        print(f"    투자금: {sizing['position_value']:,.0f}원 ({sizing['position_pct']:.1f}%)")
        print(f"    리스크 금액: {sizing['risk_amount']:,.0f}원 ({sizing['risk_pct']:.2f}%)")

    # 4. 리스크 모니터링
    print(f"\n{'═' * 60}")
    print(f"  🛡️ 리스크 모니터링 시스템")
    print(f"{'═' * 60}")
    monitor = RiskMonitor()
    generate_demo_trades(monitor, n_trades=30)
    monitor.check_portfolio_risk(
        stats.total_value, stats.total_risk,
        stats.strategy_allocation, stats.correlation_avg
    )
    print(monitor.get_dashboard(stats.total_value))

    # 5. 포트폴리오 최적화
    print(f"\n{'═' * 60}")
    print(f"  ⚡ 포트폴리오 최적화 (4가지 방법)")
    print(f"{'═' * 60}")
    optimizer = PortfolioOptimizer(returns)
    opt_results = optimizer.compare_all()
    print(optimizer.format_comparison(opt_results))

    # 6. 효율적 프론티어
    print(f"\n  ⏳ 효율적 프론티어 산출 중...")
    frontier = optimizer.efficient_frontier(n_points=50, n_portfolios=15000)
    print(f"  ✅ {len(frontier)}개 포인트 산출")

    # 7. 시각화
    print(f"\n  📊 차트 생성 중...")
    os.makedirs(output_dir, exist_ok=True)

    plot_efficient_frontier(
        frontier, opt_results,
        save_path=os.path.join(output_dir, "efficient_frontier.png")
    )
    plot_correlation_heatmap(
        returns,
        save_path=os.path.join(output_dir, "correlation.png")
    )

    # 최대 샤프 비중으로 리스크 기여도
    best_opt = max(opt_results, key=lambda x: x.sharpe_ratio)
    plot_risk_contribution(
        best_opt.weights, returns,
        save_path=os.path.join(output_dir, "risk_contribution.png")
    )

    plot_position_sizing(
        portfolio.initial_capital,
        save_path=os.path.join(output_dir, "position_sizing.png")
    )

    plot_risk_dashboard(
        stats,
        save_path=os.path.join(output_dir, "risk_dashboard.png")
    )

    print(f"\n{'═' * 60}")
    print(f"  ✅ 전체 분석 완료!")
    print(f"  📁 차트 저장: {output_dir}/")
    print(f"{'═' * 60}\n")


def run_sizing_calculator():
    """대화형 포지션 사이징 계산기"""
    print(f"\n{'═' * 50}")
    print(f"  📐 포지션 사이징 계산기")
    print(f"{'═' * 50}")

    capital = float(input("  계좌 자산 (원): ") or 100000000)
    risk_pct = float(input("  매매당 리스크 (%): ") or 1.0)
    entry = float(input("  진입 예정가: "))
    stop = float(input("  손절가: "))

    portfolio = Portfolio(initial_capital=capital, risk_per_trade_pct=risk_pct)
    result = portfolio.calculate_position_size(entry, stop)

    print(f"\n  ── 결과 ──")
    print(f"  매수 수량: {result['shares']:,}주")
    print(f"  투자금: {result['position_value']:,.0f}원 ({result['position_pct']:.1f}%)")
    print(f"  리스크 금액: {result['risk_amount']:,.0f}원 ({result['risk_pct']:.2f}%)")
    print(f"  리스크/주: {result['risk_per_share']:,.0f}원")


def main():
    parser = argparse.ArgumentParser(
        description="📊 포트폴리오 리스크 매니저 — 포지션 사이징, 최적화, 리스크 모니터링",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("command", nargs="?", default="demo",
                        choices=["demo", "sizing", "optimize"],
                        help="실행 모드:\n  demo: 종합 데모\n  sizing: 포지션 사이징 계산기\n  optimize: 최적화만 실행")
    parser.add_argument("--output", default=None, help="차트 출력 디렉토리")
    parser.add_argument("--capital", type=float, default=100_000_000, help="초기 자본")
    parser.add_argument("--risk", type=float, default=1.0, help="매매당 리스크 %%")

    args = parser.parse_args()
    output_dir = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    if args.command == "demo":
        run_demo(output_dir)
    elif args.command == "sizing":
        run_sizing_calculator()
    elif args.command == "optimize":
        print("\n  ⏳ 최적화 실행 중...")
        returns = generate_synthetic_returns()
        optimizer = PortfolioOptimizer(returns)
        results = optimizer.compare_all()
        print(optimizer.format_comparison(results))

        os.makedirs(output_dir, exist_ok=True)
        frontier = optimizer.efficient_frontier()
        plot_efficient_frontier(frontier, results,
                                save_path=os.path.join(output_dir, "efficient_frontier.png"))


if __name__ == "__main__":
    main()
