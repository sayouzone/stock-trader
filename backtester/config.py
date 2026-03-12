"""
백테스트 설정 파일
- 기간, 종목, 전략 파라미터 등
"""

# ── 데이터 설정 ──
DATA_CONFIG = {
    "start_date": "2021-01-01",
    "end_date": "2025-12-31",
    "market": "us",                    # "kr" (한국) 또는 "us" (미국)
    "num_stocks": 10,                  # 다운로드 종목 수
    "benchmark": "^GSPC",              # 벤치마크: ^KS11 (KOSPI), ^GSPC (S&P 500)

    # 직접 종목 지정 시 (이 리스트가 비어 있으면 market 기반 자동 선정)
    "tickers": [],
    # 예: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "JNJ", "BRK-B"]
    # 예: ["005930.KS", "000660.KS", "035420.KS", "005380.KS", "068270.KS"]
}

# ── 포트폴리오 설정 ──
PORTFOLIO_CONFIG = {
    "initial_capital": 100_000_000,    # 1억원
    "commission_rate": 0.00015,        # 매매 수수료 0.015%
    "tax_rate": 0.0023,                # 매도세 0.23% (한국 기준)
}

# ── 실행할 전략 목록 ──
# True로 설정된 전략만 백테스트 실행
RUN_STRATEGIES = {
    "position": True,       # 포지션 트레이딩
    "growth": True,         # 성장주 트레이딩
    "value": True,          # 가치 트레이딩
    "swing": True,          # 스윙 트레이딩
    "momentum": True,       # 모멘텀 트레이딩
    "dividend": True,       # 배당 투자
}

# ── 전략별 파라미터 커스텀 (기본값 오버라이드) ──
STRATEGY_PARAMS = {
    "position": {
        # "account_risk_pct": 0.02,
        # "atr_stop_multiplier": 3.0,
    },
    "growth": {
        # "stop_loss_pct": 0.08,
    },
    "value": {
        # "max_positions": 20,
    },
    "swing": {
        # "time_stop_days": 10,
    },
    "momentum": {
        # "rebalance_freq_days": 30,
    },
    "dividend": {
        # "buy_interval_days": 30,
    },
}

# ── 출력 설정 ──
OUTPUT_CONFIG = {
    "output_dir": "results",
    "save_charts": True,
    "save_csv": True,
    "print_trades": True,           # 개별 거래 내역 출력
    "max_trades_display": 20,       # 표시할 최대 거래 수
}
