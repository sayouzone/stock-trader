"""
종목 스코어링 엔진
- 각 스크리너가 산출한 항목별 점수를 합산하여 종합 점수 산출
- 조건 통과/미통과를 체크리스트로 표시
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Grade(Enum):
    STRONG_BUY = "★★★★★"
    BUY = "★★★★☆"
    WATCH = "★★★☆☆"
    NEUTRAL = "★★☆☆☆"
    AVOID = "★☆☆☆☆"

    @classmethod
    def from_score(cls, score: float, max_score: float) -> "Grade":
        pct = score / max_score if max_score > 0 else 0
        if pct >= 0.80:
            return cls.STRONG_BUY
        elif pct >= 0.65:
            return cls.BUY
        elif pct >= 0.50:
            return cls.WATCH
        elif pct >= 0.30:
            return cls.NEUTRAL
        else:
            return cls.AVOID


@dataclass
class CheckItem:
    """개별 체크 항목"""
    name: str
    passed: bool
    value: str  # 표시용 값
    weight: float = 1.0  # 가중치

    @property
    def score(self) -> float:
        return self.weight if self.passed else 0.0


@dataclass
class ScreenResult:
    """단일 종목의 스크리닝 결과"""
    ticker: str
    strategy: str
    checks: list[CheckItem] = field(default_factory=list)
    entry_signals: list[str] = field(default_factory=list)
    stop_loss: float | None = None
    target_price: float | None = None
    current_price: float = 0.0

    @property
    def total_score(self) -> float:
        return sum(c.score for c in self.checks)

    @property
    def max_score(self) -> float:
        return sum(c.weight for c in self.checks)

    @property
    def score_pct(self) -> float:
        return self.total_score / self.max_score * 100 if self.max_score > 0 else 0

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total_count(self) -> int:
        return len(self.checks)

    @property
    def grade(self) -> Grade:
        return Grade.from_score(self.total_score, self.max_score)

    def summary_row(self) -> dict:
        """테이블 행 데이터"""
        return {
            "종목": self.ticker,
            "전략": self.strategy,
            "점수": f"{self.total_score:.1f}/{self.max_score:.1f}",
            "통과": f"{self.passed_count}/{self.total_count}",
            "등급": self.grade.value,
            "현재가": f"{self.current_price:,.0f}",
            "손절가": f"{self.stop_loss:,.0f}" if self.stop_loss else "-",
            "시그널": " | ".join(self.entry_signals) if self.entry_signals else "-",
        }

    def detail_str(self) -> str:
        """상세 체크리스트 문자열"""
        lines = [
            f"\n{'─' * 55}",
            f"  {self.ticker}  │  {self.strategy}  │  {self.grade.value}  ({self.score_pct:.0f}점)",
            f"{'─' * 55}",
        ]
        for c in self.checks:
            icon = "✅" if c.passed else "❌"
            w = f"(×{c.weight:.1f})" if c.weight != 1.0 else ""
            lines.append(f"  {icon} {c.name:24s} {c.value:>12s} {w}")

        if self.entry_signals:
            lines.append(f"\n  📌 진입 시그널:")
            for sig in self.entry_signals:
                lines.append(f"     → {sig}")

        if self.stop_loss:
            lines.append(f"  🛑 손절가: {self.stop_loss:,.0f}")
        if self.target_price:
            lines.append(f"  🎯 목표가: {self.target_price:,.0f}")

        return "\n".join(lines)
