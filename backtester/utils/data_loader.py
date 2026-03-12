"""
데이터 로드 모듈
- yfinance를 이용한 주가 데이터 다운로드
- 한국 주식 (KRX) 및 미국 주식 지원
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataLoader:
    """주가 데이터 로더"""

    # 주요 한국 주식 (코스피 대형주) 예시 티커
    KRX_EXAMPLES = {
        "삼성전자": "005930.KS",
        "SK하이닉스": "000660.KS",
        "LG에너지솔루션": "373220.KS",
        "삼성바이오로직스": "207940.KS",
        "현대자동차": "005380.KS",
        "기아": "000270.KS",
        "셀트리온": "068270.KS",
        "KB금융": "105560.KS",
        "POSCO홀딩스": "005490.KS",
        "NAVER": "035420.KS",
        "카카오": "035720.KS",
        "LG화학": "051910.KS",
        "삼성SDI": "006400.KS",
        "현대모비스": "012330.KS",
        "신한지주": "055550.KS",
        "하나금융지주": "086790.KS",
        "LG전자": "066570.KS",
        "삼성물산": "028260.KS",
        "한국전력": "015760.KS",
        "KT&G": "033780.KS",
    }

    # 미국 주요 주식 예시
    US_EXAMPLES = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Google": "GOOGL",
        "Amazon": "AMZN",
        "NVIDIA": "NVDA",
        "Meta": "META",
        "Tesla": "TSLA",
        "Berkshire": "BRK-B",
        "JPMorgan": "JPM",
        "JNJ": "JNJ",
    }

    @staticmethod
    def download(
        tickers: list[str],
        start: str = "2020-01-01",
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """
        yfinance로 주가 데이터 다운로드

        Args:
            tickers: 종목 코드 리스트 (예: ["005930.KS", "AAPL"])
            start: 시작일 (YYYY-MM-DD)
            end: 종료일 (기본값: 오늘)
            interval: 데이터 간격 (1d, 1wk, 1mo)

        Returns:
            {ticker: DataFrame} 딕셔너리
        """
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        data = {}
        for ticker in tickers:
            try:
                logger.info(f"데이터 다운로드: {ticker}")
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                )

                if df.empty:
                    logger.warning(f"데이터 없음: {ticker}")
                    continue

                # 멀티인덱스 컬럼 정리
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # NaN 제거
                df = df.dropna()

                if len(df) > 0:
                    data[ticker] = df
                    logger.info(
                        f"  {ticker}: {len(df)}일 ({df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d})"
                    )

            except Exception as e:
                logger.error(f"다운로드 실패 {ticker}: {e}")

        return data

    @staticmethod
    def get_benchmark(
        ticker: str = "^KS11",
        start: str = "2020-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        벤치마크 지수 다운로드

        티커:
            ^KS11: KOSPI
            ^KQ11: KOSDAQ
            ^GSPC: S&P 500
            ^IXIC: NASDAQ
        """
        data = DataLoader.download([ticker], start=start, end=end)
        return data.get(ticker, pd.DataFrame())

    @staticmethod
    def get_korean_stocks(
        n: int = 10,
        start: str = "2020-01-01",
        end: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """한국 주요 주식 N개 다운로드"""
        tickers = list(DataLoader.KRX_EXAMPLES.values())[:n]
        return DataLoader.download(tickers, start=start, end=end)

    @staticmethod
    def get_us_stocks(
        n: int = 10,
        start: str = "2020-01-01",
        end: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """미국 주요 주식 N개 다운로드"""
        tickers = list(DataLoader.US_EXAMPLES.values())[:n]
        return DataLoader.download(tickers, start=start, end=end)
