"""
리스크 모니터링 시스템
- 문서의 리스크 관리 규칙을 실시간 추적
  ① 3연패 → 포지션 사이즈 50% 축소
  ② 5연패 → 매매 중단
  ③ 월간 손실 -6% → 해당 월 매매 중단
- 포트폴리오 수준 리스크 한도 모니터링
- 경고/경보 시스템
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from portfolio import Strategy


class AlertLevel(Enum):
    INFO = ("ℹ️", "정보")
    WARNING = ("⚠️", "주의")
    DANGER = ("🚨", "위험")
    CRITICAL = ("🔴", "긴급")

    def __init__(self, icon: str, label: str):
        self.icon = icon
        self._label = label


@dataclass
class Alert:
    level: AlertLevel
    rule: str
    message: str
    action: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradeRecord:
    """매매 기록 (리스크 추적용)"""
    date: date
    strategy: Strategy
    ticker: str
    pnl: float
    r_multiple: float
    is_win: bool


class RiskMonitor:
    """리스크 모니터링 엔진"""

    # 문서 기반 규칙 임계값
    CONSECUTIVE_LOSS_WARNING = 3   # 3연패 → 50% 축소
    CONSECUTIVE_LOSS_STOP = 5      # 5연패 → 매매 중단
    MONTHLY_LOSS_LIMIT = -0.06     # 월간 -6%
    MAX_PORTFOLIO_RISK = 0.06      # 포트폴리오 총 리스크 6%
    MAX_SINGLE_RISK = 0.02         # 단일 포지션 리스크 2%
    MAX_STRATEGY_PCT = 0.30        # 전략별 최대 비중 30%
    MAX_CORRELATION = 0.70         # 상관관계 경고 임계

    def __init__(self):
        self.trade_history: list[TradeRecord] = []
        self.alerts: list[Alert] = []
        self.is_trading_halted = False
        self.risk_reduction_factor = 1.0  # 1.0 = 정상, 0.5 = 50% 축소

    def add_trade(self, record: TradeRecord):
        self.trade_history.append(record)
        self._check_consecutive_losses()
        self._check_monthly_loss()

    # ──────────────────────────────────────────
    #  규칙 ① 연패 감지
    # ──────────────────────────────────────────
    def _check_consecutive_losses(self):
        if not self.trade_history:
            return

        consec = 0
        for t in reversed(self.trade_history):
            if not t.is_win:
                consec += 1
            else:
                break

        if consec >= self.CONSECUTIVE_LOSS_STOP:
            self.is_trading_halted = True
            self.risk_reduction_factor = 0.0
            self.alerts.append(Alert(
                AlertLevel.CRITICAL,
                "5연패 규칙",
                f"연속 {consec}회 손실 발생 — 매매 즉시 중단",
                "전략 점검 후 데모 트레이딩으로 복귀. 최소 2주 휴식 권장."
            ))
        elif consec >= self.CONSECUTIVE_LOSS_WARNING:
            self.risk_reduction_factor = 0.5
            self.alerts.append(Alert(
                AlertLevel.DANGER,
                "3연패 규칙",
                f"연속 {consec}회 손실 — 포지션 사이즈 50% 축소",
                "리스크 금액을 절반으로 줄이고 진입 기준을 엄격하게 적용."
            ))
        elif consec == 2:
            self.alerts.append(Alert(
                AlertLevel.WARNING,
                "연패 주의",
                f"연속 {consec}회 손실 — 3연패 임박 주의",
                "다음 매매 전 전략 조건 재확인. 무리한 진입 자제."
            ))
        else:
            # 연패 해소 시 복원
            if self.risk_reduction_factor < 1.0 and not self.is_trading_halted:
                self.risk_reduction_factor = 1.0
                self.alerts.append(Alert(
                    AlertLevel.INFO,
                    "연패 해소",
                    "승리로 연패 탈출 — 포지션 사이즈 정상 복귀",
                    "정상 리스크 비율로 복귀."
                ))

    # ──────────────────────────────────────────
    #  규칙 ② 월간 손실 한도
    # ──────────────────────────────────────────
    def _check_monthly_loss(self):
        today = date.today()
        month_trades = [t for t in self.trade_history
                        if t.date.year == today.year and t.date.month == today.month]
        if not month_trades:
            return

        month_pnl = sum(t.pnl for t in month_trades)
        # 간이 월간 수익률 (초기 자본 기준)
        month_return = month_pnl / 100_000_000  # 기본 1억 기준

        if month_return <= self.MONTHLY_LOSS_LIMIT:
            self.is_trading_halted = True
            self.alerts.append(Alert(
                AlertLevel.CRITICAL,
                "월간 손실 한도",
                f"이번 달 손실 {month_return:.1%} — 한도 {self.MONTHLY_LOSS_LIMIT:.0%} 초과",
                "이번 달 잔여 기간 매매 중단. 다음 달 첫 거래일에 재개."
            ))
        elif month_return <= self.MONTHLY_LOSS_LIMIT * 0.7:
            self.alerts.append(Alert(
                AlertLevel.DANGER,
                "월간 손실 경고",
                f"이번 달 손실 {month_return:.1%} — 한도의 70% 소진",
                "매매 빈도 축소. 확실한 시그널에만 진입."
            ))

    # ──────────────────────────────────────────
    #  포트폴리오 수준 리스크 체크
    # ──────────────────────────────────────────
    def check_portfolio_risk(self, total_value: float, total_risk: float,
                              strategy_allocation: dict,
                              correlation_avg: float = 0) -> list[Alert]:
        """포트폴리오 수준 리스크 점검"""
        new_alerts = []

        # 총 리스크
        risk_pct = total_risk / total_value if total_value > 0 else 0
        if risk_pct > self.MAX_PORTFOLIO_RISK:
            new_alerts.append(Alert(
                AlertLevel.DANGER,
                "포트폴리오 리스크 초과",
                f"총 리스크 {risk_pct:.1%} > 한도 {self.MAX_PORTFOLIO_RISK:.0%}",
                "가장 리스크가 높은 포지션부터 축소 또는 청산."
            ))
        elif risk_pct > self.MAX_PORTFOLIO_RISK * 0.8:
            new_alerts.append(Alert(
                AlertLevel.WARNING,
                "포트폴리오 리스크 주의",
                f"총 리스크 {risk_pct:.1%} — 한도 접근 중",
                "신규 포지션 추가 전 기존 리스크 재검토."
            ))

        # 전략 집중 리스크
        for strat, pct in strategy_allocation.items():
            if pct / 100 > self.MAX_STRATEGY_PCT:
                new_alerts.append(Alert(
                    AlertLevel.WARNING,
                    "전략 집중 리스크",
                    f"{strat} 전략 비중 {pct:.1f}% > 한도 {self.MAX_STRATEGY_PCT * 100:.0f}%",
                    f"{strat} 포지션 축소 또는 다른 전략으로 분산."
                ))

        # 상관관계 리스크
        if correlation_avg > self.MAX_CORRELATION:
            new_alerts.append(Alert(
                AlertLevel.WARNING,
                "상관관계 리스크",
                f"평균 상관계수 {correlation_avg:.2f} > 경고 임계 {self.MAX_CORRELATION}",
                "저상관 자산 편입으로 분산 효과 개선 필요."
            ))

        self.alerts.extend(new_alerts)
        return new_alerts

    # ──────────────────────────────────────────
    #  리스크 현황 대시보드
    # ──────────────────────────────────────────
    def get_dashboard(self, total_value: float = 0) -> str:
        lines = [
            f"\n{'═' * 60}",
            f"  🛡️ 리스크 모니터링 대시보드",
            f"{'═' * 60}",
        ]

        # 매매 상태
        if self.is_trading_halted:
            lines.append(f"  🔴 매매 상태: 중단됨")
        elif self.risk_reduction_factor < 1.0:
            lines.append(f"  🟡 매매 상태: 축소 (사이즈 {self.risk_reduction_factor:.0%})")
        else:
            lines.append(f"  🟢 매매 상태: 정상")

        lines.append(f"  리스크 축소 계수: {self.risk_reduction_factor:.0%}")

        # 연패 현황
        consec = 0
        for t in reversed(self.trade_history):
            if not t.is_win:
                consec += 1
            else:
                break
        lines.append(f"  현재 연패: {consec}회 (3연패 주의, 5연패 중단)")

        # 월간 손익
        today = date.today()
        month_trades = [t for t in self.trade_history
                        if t.date.year == today.year and t.date.month == today.month]
        month_pnl = sum(t.pnl for t in month_trades)
        month_return = month_pnl / total_value if total_value > 0 else 0
        lines.append(f"  이번 달 손익: {month_pnl:+,.0f}원 ({month_return:+.2%}) / 한도 -6%")

        # 매매 통계
        if self.trade_history:
            total = len(self.trade_history)
            wins = sum(1 for t in self.trade_history if t.is_win)
            avg_r = sum(t.r_multiple for t in self.trade_history) / total
            lines.extend([
                f"\n{'─' * 60}",
                f"  📊 매매 통계 (총 {total}건)",
                f"{'─' * 60}",
                f"  승률: {wins / total:.1%} ({wins}승 {total - wins}패)",
                f"  평균 R배수: {avg_r:+.2f}R",
            ])

            # 전략별 통계
            strat_stats = {}
            for t in self.trade_history:
                key = t.strategy.label
                if key not in strat_stats:
                    strat_stats[key] = {"wins": 0, "total": 0, "pnl": 0, "r_sum": 0}
                strat_stats[key]["total"] += 1
                strat_stats[key]["pnl"] += t.pnl
                strat_stats[key]["r_sum"] += t.r_multiple
                if t.is_win:
                    strat_stats[key]["wins"] += 1

            lines.append(f"\n  {'전략':8s} {'승률':>8s} {'매매':>5s} {'손익':>12s} {'평균R':>7s}")
            for strat, s in sorted(strat_stats.items(), key=lambda x: -x[1]["pnl"]):
                wr = s["wins"] / s["total"] if s["total"] > 0 else 0
                avg_r = s["r_sum"] / s["total"]
                lines.append(
                    f"  {strat:8s} {wr:>7.0%} {s['total']:>5d} {s['pnl']:>+12,.0f} {avg_r:>+7.2f}R"
                )

        # 최근 경고
        recent = self.alerts[-5:] if self.alerts else []
        if recent:
            lines.extend([
                f"\n{'─' * 60}",
                f"  🔔 최근 경고 ({len(self.alerts)}건 중 최근 5건)",
                f"{'─' * 60}",
            ])
            for a in reversed(recent):
                lines.append(f"  {a.level.icon} [{a.rule}] {a.message}")
                lines.append(f"     → {a.action}")

        return "\n".join(lines)
