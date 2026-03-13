"""
시장 레짐 감지 엔진
- 다중 지표를 결합하여 현재 시장 국면(강세/약세/횡보)을 판별
- 레짐 전환 시점을 감지하고 이력을 추적
- 문서의 시장 환경 판단 기준 반영:
  ① 200MA 위/아래 (포지션 전략의 시장 필터)
  ② ADX 추세 강도
  ③ MA 배열 상태
  ④ RSI 레벨
  ⑤ MACD 방향성
  ⑥ 변동성 수준 (ATR, BB Width)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd


class Regime(Enum):
    STRONG_BULL = ("🟢 강한 강세", "strong_bull")
    BULL = ("🔵 강세", "bull")
    SIDEWAYS = ("🟡 횡보", "sideways")
    BEAR = ("🔴 약세", "bear")
    STRONG_BEAR = ("⚫ 강한 약세", "strong_bear")

    def __init__(self, label: str, key: str):
        self.label = label
        self.key = key


@dataclass
class RegimeSignal:
    """개별 레짐 판별 시그널"""
    name: str
    value: float
    regime_score: float  # -2(강한약세) ~ +2(강한강세)
    description: str


@dataclass
class RegimeResult:
    """레짐 판별 결과"""
    date: pd.Timestamp
    regime: Regime
    confidence: float  # 0~100%
    composite_score: float  # -2 ~ +2
    signals: list[RegimeSignal] = field(default_factory=list)
    close_price: float = 0.0

    def summary(self) -> str:
        lines = [
            f"  날짜: {self.date.strftime('%Y-%m-%d')}",
            f"  종가: {self.close_price:,.0f}",
            f"  레짐: {self.regime.label}",
            f"  확신도: {self.confidence:.0f}%",
            f"  종합 점수: {self.composite_score:+.2f} (-2.0 약세 ~ +2.0 강세)",
        ]
        return "\n".join(lines)


class RegimeDetector:
    """
    다중 지표 기반 시장 레짐 감지기

    6가지 시그널을 가중 합산하여 -2 ~ +2 범위의 종합 점수를 산출.
    점수 구간별로 5단계 레짐으로 분류.
    """

    # 레짐 분류 임계값
    THRESHOLDS = {
        "strong_bull": 1.2,
        "bull": 0.4,
        "sideways_upper": 0.4,
        "sideways_lower": -0.4,
        "bear": -1.2,
    }

    def detect_single(self, df: pd.DataFrame) -> RegimeResult:
        """현재(마지막 행) 레짐 판별"""
        if len(df) < 200:
            return RegimeResult(
                date=df.index[-1], regime=Regime.SIDEWAYS,
                confidence=0, composite_score=0.0, close_price=df["Close"].iloc[-1]
            )

        last = df.iloc[-1]
        signals = []

        # ── 1. 200MA 필터 (가중치 높음) ──
        sma200 = last.get("SMA_200")
        if not pd.isna(sma200) and sma200 > 0:
            pct_above = (last["Close"] - sma200) / sma200
            if pct_above > 0.10:
                score = 2.0
                desc = f"200MA 위 {pct_above:+.1%} (강세)"
            elif pct_above > 0.02:
                score = 1.0
                desc = f"200MA 위 {pct_above:+.1%}"
            elif pct_above > -0.02:
                score = 0.0
                desc = f"200MA 근접 {pct_above:+.1%}"
            elif pct_above > -0.10:
                score = -1.0
                desc = f"200MA 아래 {pct_above:+.1%}"
            else:
                score = -2.0
                desc = f"200MA 대폭 하회 {pct_above:+.1%} (약세)"
            signals.append(RegimeSignal("200MA 필터", pct_above, score, desc))

        # ── 2. MA 배열 상태 ──
        sma50 = last.get("SMA_50")
        sma150 = last.get("SMA_150")
        if all(not pd.isna(v) for v in [sma50, sma150, sma200]):
            if sma50 > sma150 > sma200:
                score = 2.0
                desc = "정배열 (50>150>200)"
            elif sma50 > sma200:
                score = 1.0
                desc = "부분 정배열"
            elif sma50 < sma150 < sma200:
                score = -2.0
                desc = "역배열 (50<150<200)"
            elif sma50 < sma200:
                score = -1.0
                desc = "부분 역배열"
            else:
                score = 0.0
                desc = "MA 혼조"
            signals.append(RegimeSignal("MA 배열", 0, score, desc))

        # ── 3. ADX 추세 강도 + 방향 ──
        adx_val = last.get("ADX", 0)
        if not pd.isna(adx_val):
            # ADX는 추세 강도만 나타냄 → 방향은 가격 위치로 결합
            price_above_ma = last["Close"] > sma200 if not pd.isna(sma200) else True
            if adx_val > 30:
                score = 1.5 if price_above_ma else -1.5
                desc = f"강한 추세 ADX={adx_val:.0f} ({'상승' if price_above_ma else '하락'})"
            elif adx_val > 20:
                score = 0.5 if price_above_ma else -0.5
                desc = f"추세 존재 ADX={adx_val:.0f}"
            else:
                score = 0.0
                desc = f"추세 약함 ADX={adx_val:.0f} (횡보)"
            signals.append(RegimeSignal("ADX 추세 강도", adx_val, score, desc))

        # ── 4. RSI 영역 ──
        rsi_val = last.get("RSI")
        if not pd.isna(rsi_val):
            if rsi_val > 70:
                score = 1.5
                desc = f"RSI={rsi_val:.0f} 과매수 (강세 과열)"
            elif rsi_val > 55:
                score = 1.0
                desc = f"RSI={rsi_val:.0f} 강세 영역"
            elif rsi_val > 45:
                score = 0.0
                desc = f"RSI={rsi_val:.0f} 중립"
            elif rsi_val > 30:
                score = -1.0
                desc = f"RSI={rsi_val:.0f} 약세 영역"
            else:
                score = -1.5
                desc = f"RSI={rsi_val:.0f} 과매도 (약세 극단)"
            signals.append(RegimeSignal("RSI 레벨", rsi_val, score, desc))

        # ── 5. MACD 방향성 ──
        macd_hist = last.get("MACD_Hist")
        macd_val = last.get("MACD")
        if not pd.isna(macd_hist) and not pd.isna(macd_val):
            if macd_val > 0 and macd_hist > 0:
                score = 1.5
                desc = "MACD 양(+) + 히스토그램 확대"
            elif macd_val > 0:
                score = 0.5
                desc = "MACD 양(+), 모멘텀 둔화"
            elif macd_val < 0 and macd_hist < 0:
                score = -1.5
                desc = "MACD 음(-) + 히스토그램 확대"
            elif macd_val < 0:
                score = -0.5
                desc = "MACD 음(-), 하락 둔화"
            else:
                score = 0.0
                desc = "MACD 중립"
            signals.append(RegimeSignal("MACD 방향", macd_val, score, desc))

        # ── 6. 변동성 수준 (BB Width) ──
        bb_width = last.get("BB_Width")
        vol_21 = last.get("Volatility_21d")
        if not pd.isna(bb_width):
            # 높은 변동성 = 불확실 → 레짐 판별에 변동성 페널티/보너스 적용
            if bb_width > 0.15:
                score = -0.5  # 극단적 변동성 → 약간 부정적
                desc = f"BB Width={bb_width:.3f} 고변동성 (불확실)"
            elif bb_width < 0.04:
                score = 0.5  # 저변동성 = 스퀴즈 → 돌파 대기
                desc = f"BB Width={bb_width:.3f} 저변동성 (스퀴즈)"
            else:
                score = 0.0
                desc = f"BB Width={bb_width:.3f} 정상 범위"
            signals.append(RegimeSignal("변동성", bb_width, score, desc))

        # ── 종합 점수 산출 (가중 평균) ──
        weights = [2.0, 1.5, 1.5, 1.0, 1.0, 0.5]  # 200MA가 가장 중요
        if len(signals) < len(weights):
            weights = weights[:len(signals)]
        total_w = sum(weights)
        composite = sum(s.regime_score * w for s, w in zip(signals, weights)) / total_w if total_w > 0 else 0.0

        # ── 레짐 분류 ──
        if composite >= self.THRESHOLDS["strong_bull"]:
            regime = Regime.STRONG_BULL
        elif composite >= self.THRESHOLDS["bull"]:
            regime = Regime.BULL
        elif composite >= self.THRESHOLDS["sideways_lower"]:
            regime = Regime.SIDEWAYS
        elif composite >= self.THRESHOLDS["bear"]:
            regime = Regime.BEAR
        else:
            regime = Regime.STRONG_BEAR

        # 확신도: 극단에 가까울수록 높음
        confidence = min(abs(composite) / 2.0 * 100, 100)

        return RegimeResult(
            date=df.index[-1],
            regime=regime,
            confidence=confidence,
            composite_score=composite,
            signals=signals,
            close_price=last["Close"],
        )

    def detect_history(self, df: pd.DataFrame, window: int = 200) -> list[RegimeResult]:
        """전체 기간의 레짐 이력 산출 (window일 이후부터)"""
        results = []
        start = max(window, 200)
        for i in range(start, len(df)):
            sub = df.iloc[:i + 1]
            result = self.detect_single(sub)
            results.append(result)
        return results

    def find_transitions(self, history: list[RegimeResult]) -> list[dict]:
        """레짐 전환 시점 탐지"""
        transitions = []
        for i in range(1, len(history)):
            if history[i].regime != history[i - 1].regime:
                transitions.append({
                    "date": history[i].date,
                    "from": history[i - 1].regime,
                    "to": history[i].regime,
                    "score_before": history[i - 1].composite_score,
                    "score_after": history[i].composite_score,
                    "price": history[i].close_price,
                })
        return transitions
