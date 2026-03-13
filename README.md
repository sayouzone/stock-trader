# Stock Trader

Stock Trading agents built by Claude Code

## 구성

1. 실시간 종목 스크리너
2. 포트폴리오 리스크 매니저
포토폴리오 최적화기
리스크 관리 대시보드
3. 백테스팅 시뮬레이터
4. Streamlit 대시보드
5. 매매일지 & 복기 시스템
6. 전략 학습 퀴즈 앱
시장 레짐 감지기

### Screener (실시간 종목 스크리너)

```
screener/
├── screener.py              # CLI 메인 (--demo, --tickers, --strategies, --detail, --synthetic)
├── indicators.py            # 20+ 기술적 지표 일괄 계산
├── scoring.py               # 가중 점수 + 등급(★) 시스템
├── requirements.txt
├── screeners/
    ├── base.py              # 전략 추상 클래스
    ├── position.py          # 포지션 트레이딩 (200MA, ADX, Stage 2)
    ├── growth.py            # 성장주 (CAN SLIM, 피봇 돌파, MA 정배열)
    ├── value.py             # 가치 투자 (52주 저점, 골든크로스, MACD 0선)
    ├── swing.py             # 스윙 (RSI 반전, 스토캐스틱, BB 반등)
    ├── momentum.py          # 모멘텀 (듀얼 모멘텀, 복합 스코어)
    └── dividend.py          # 배당 (저변동성, 가격밴드 하단, DCA)
```

### Backtester (백테스팅 시뮬레이터)

```
backtester/
├── main.py                  # 메인 실행 (yfinance로 실제 데이터)
├── test_with_synthetic.py   # 합성 데이터 테스트
├── config.py                # 설정 (기간, 종목, 전략 파라미터)
├── engine.py                # 백테스팅 엔진 (Portfolio, Trade, Engine)
├── requirements.txt
├── strategies/
│   ├── base.py              # 전략 추상 클래스
│   ├── position.py          # 포지션 트레이딩 (200MA, ATR 손절)
│   ├── growth.py            # 성장주 트레이딩 (피벗 돌파, 7-8% 손절)
│   ├── value.py             # 가치 트레이딩 (골든크로스, MACD 0선)
│   ├── swing.py             # 스윙 트레이딩 (RSI, 시간 손절)
│   ├── momentum.py          # 모멘텀 트레이딩 (듀얼 모멘텀, 리밸런싱)
│   └── dividend.py          # 배당 투자 (DCA, 배당률 밴드)
├── indicators/
│   ├── technical.py         # 기술적 지표 20종+ (SMA/EMA/RSI/MACD/ATR/ADX...)
│   └── fundamental.py       # 펀더멘털 지표 (PER/PBR/ROE/F-Score/마법공식)
└── utils/
    ├── data_loader.py       # yfinance 데이터 로더 (한국/미국)
    ├── performance.py       # 성과 분석 (CAGR/MDD/샤프/승률/목표대비)
    └── visualizer.py        # 시각화 (수익곡선/드로다운/전략비교)
```

### 매매일지 & 복기 시스템

```
trading_journal/
├── journal.py      # CLI 메인 (add/close/list/open/report/stats/note/demo)
├── db.py           # SQLite DB 스키마 + CRUD
├── analytics.py    # 성과분석 + 복기 리포트 엔진
└── requirements.txt
```

##

### Screener (실시간 종목 스크리너)

```bash
python screener.py --demo                    # 데모 모드
python screener.py -t AAPL MSFT --detail     # 특정 종목 분석
python screener.py -s swing momentum         # 특정 전략만
```