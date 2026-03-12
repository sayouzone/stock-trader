"""
성과 분석 및 복기 리포트 엔진
- 문서의 기법별 승률/손익비 목표 대비 실적 분석
- 심리 패턴, 실수 분류, 개선 포인트 자동 도출
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime

import db

# 문서의 기법별 목표치
TARGETS = {
    "포지션": {"win_rate": (40, 50), "rr": 3.0, "holding": "수주~수개월"},
    "성장주": {"win_rate": (35, 45), "rr": 3.0, "holding": "수주~수년"},
    "가치":   {"win_rate": (55, 65), "rr": 2.0, "holding": "수개월~수년"},
    "스윙":   {"win_rate": (50, 60), "rr": 2.0, "holding": "수일~수주"},
    "모멘텀": {"win_rate": (45, 55), "rr": 2.0, "holding": "수일~수주"},
    "배당":   {"win_rate": (65, 75), "rr": 1.5, "holding": "수년 이상"},
}


def _safe_pct(num: int, denom: int) -> float:
    return num / denom * 100 if denom > 0 else 0.0


def compute_stats(trades: list[dict]) -> dict:
    """거래 목록으로부터 핵심 통계 계산"""
    if not trades:
        return {"count": 0}

    closed = [t for t in trades if t.get("exit_date")]
    open_trades = [t for t in trades if not t.get("exit_date")]

    winners = [t for t in closed if (t.get("pnl") or 0) > 0]
    losers = [t for t in closed if (t.get("pnl") or 0) <= 0]

    total = len(closed)
    win_rate = _safe_pct(len(winners), total)

    avg_win = sum(t["pnl_pct"] for t in winners) / len(winners) if winners else 0
    avg_loss = abs(sum(t["pnl_pct"] for t in losers) / len(losers)) if losers else 0
    risk_reward = avg_win / avg_loss if avg_loss > 0 else float("inf")

    # 기대값: (승률×평균이익) - (패율×평균손실)
    expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)

    # R 배수 분석
    r_values = [t["r_multiple"] for t in closed if t.get("r_multiple") is not None]
    avg_r = sum(r_values) / len(r_values) if r_values else 0

    # 연승/연패
    streaks = _calc_streaks(closed)

    # 보유일
    hold_days = [t["holding_days"] for t in closed if t.get("holding_days")]
    avg_hold = sum(hold_days) / len(hold_days) if hold_days else 0

    # 계획 준수율
    plan_trades = [t for t in closed if t.get("plan_followed") is not None]
    plan_rate = _safe_pct(
        sum(1 for t in plan_trades if t["plan_followed"]),
        len(plan_trades),
    )

    # 총 손익
    total_pnl = sum(t.get("pnl", 0) or 0 for t in closed)

    return {
        "count": total,
        "open_count": len(open_trades),
        "wins": len(winners),
        "losses": len(losers),
        "win_rate": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "risk_reward": risk_reward,
        "expectancy": expectancy,
        "avg_r": avg_r,
        "max_win_streak": streaks["max_win"],
        "max_loss_streak": streaks["max_loss"],
        "current_streak": streaks["current"],
        "avg_holding_days": avg_hold,
        "plan_follow_rate": plan_rate,
        "total_pnl": total_pnl,
    }


def _calc_streaks(closed: list[dict]) -> dict:
    """연승/연패 계산"""
    if not closed:
        return {"max_win": 0, "max_loss": 0, "current": 0}

    sorted_trades = sorted(closed, key=lambda t: t.get("exit_date", ""))
    results = [1 if (t.get("pnl") or 0) > 0 else -1 for t in sorted_trades]

    max_win = max_loss = 0
    current = 0

    for r in results:
        if r > 0:
            current = max(current + 1, 1) if current > 0 else 1
            max_win = max(max_win, current)
        else:
            current = min(current - 1, -1) if current < 0 else -1
            max_loss = max(max_loss, abs(current))

    return {"max_win": max_win, "max_loss": max_loss, "current": current}


def strategy_breakdown(trades: list[dict]) -> dict[str, dict]:
    """전략별 성과 분류"""
    by_strategy = {}
    for t in trades:
        s = t.get("strategy", "기타")
        by_strategy.setdefault(s, []).append(t)

    return {s: compute_stats(ts) for s, ts in by_strategy.items()}


def emotion_analysis(trades: list[dict]) -> dict:
    """감정 패턴 분석"""
    closed = [t for t in trades if t.get("exit_date")]
    if not closed:
        return {}

    entry_emotions = Counter()
    exit_emotions = Counter()
    emotion_pnl = {}  # 진입 감정별 평균 수익

    for t in closed:
        em = t.get("entry_emotion", "")
        if em:
            entry_emotions[em] += 1
            emotion_pnl.setdefault(em, []).append(t.get("pnl_pct", 0) or 0)

        ex_em = t.get("exit_emotion", "")
        if ex_em:
            exit_emotions[ex_em] += 1

    emotion_avg = {
        em: sum(pnls) / len(pnls)
        for em, pnls in emotion_pnl.items()
    }

    return {
        "entry_emotions": dict(entry_emotions.most_common()),
        "exit_emotions": dict(exit_emotions.most_common()),
        "emotion_avg_pnl": emotion_avg,
    }


def mistake_analysis(trades: list[dict]) -> list[dict]:
    """실수 패턴 분석"""
    closed = [t for t in trades if t.get("exit_date")]
    mistakes = []

    for t in closed:
        issues = []
        pnl = t.get("pnl_pct", 0) or 0

        # 계획 미준수
        if not t.get("plan_followed"):
            issues.append("계획 미준수")

        # 추격매수 (진입 사유가 없는데 손실)
        if not t.get("entry_reason") and pnl < 0:
            issues.append("진입 근거 불명확")

        # 손절 없이 큰 손실
        if not t.get("stop_loss") and pnl < -5:
            issues.append("손절가 미설정")

        # 조급한 진입 (당일 손절)
        if t.get("holding_days") is not None and t["holding_days"] <= 1 and pnl < 0:
            issues.append("조급한 진입 (당일 손절)")

        # 감정적 매매 (불안/조급/탐욕에서 진입 + 손실)
        if t.get("entry_emotion") in ["불안", "조급", "탐욕"] and pnl < 0:
            issues.append(f"감정적 진입 ({t['entry_emotion']})")

        # 이익 손실 전환 (r_multiple이 양이었다가 음으로)
        if t.get("r_multiple") is not None and t["r_multiple"] < -0.5:
            if t.get("exit_reason") and "손절" not in t["exit_reason"]:
                issues.append("수익 → 손실 전환 (트레일링 미적용?)")

        if issues:
            mistakes.append({
                "trade_id": t["id"],
                "ticker": t["ticker"],
                "pnl_pct": pnl,
                "issues": issues,
            })

    return mistakes


def generate_report(
    start: str = None,
    end: str = None,
    strategy: str = None,
) -> str:
    """복기 리포트 생성"""
    trades = db.get_closed_trades(start=start, end=end, strategy=strategy, limit=9999)
    open_trades = db.get_open_trades()

    if not trades and not open_trades:
        return "기록된 거래가 없습니다."

    stats = compute_stats(trades + open_trades)
    lines = []

    # 헤더
    period = ""
    if start and end:
        period = f"{start} ~ {end}"
    elif start:
        period = f"{start} ~"
    elif end:
        period = f"~ {end}"
    else:
        period = "전체 기간"

    title = f"복기 리포트"
    if strategy:
        title += f" [{strategy}]"
    title += f"  ({period})"

    lines.append(f"\n{'═' * 55}")
    lines.append(f"  {title}")
    lines.append(f"{'═' * 55}")

    # ── 종합 성과 ──
    lines.append(f"\n  ■ 종합 성과")
    lines.append(f"  {'─' * 50}")
    lines.append(f"  총 거래      : {stats['count']}건 (미청산 {stats['open_count']}건)")
    lines.append(f"  승/패        : {stats['wins']}승 {stats['losses']}패")
    lines.append(f"  승률         : {stats['win_rate']:.1f}%")
    lines.append(f"  평균 수익    : +{stats['avg_win_pct']:.2f}%")
    lines.append(f"  평균 손실    : -{stats['avg_loss_pct']:.2f}%")
    lines.append(f"  손익비       : {stats['risk_reward']:.2f}")
    lines.append(f"  기대값       : {stats['expectancy']:+.2f}%")
    lines.append(f"  평균 R배수   : {stats['avg_r']:.2f}R")
    lines.append(f"  총 손익      : {stats['total_pnl']:+,.0f}원")
    lines.append(f"  평균 보유일  : {stats['avg_holding_days']:.1f}일")
    lines.append(f"  계획 준수율  : {stats['plan_follow_rate']:.0f}%")
    lines.append(f"  최대 연승    : {stats['max_win_streak']}회")
    lines.append(f"  최대 연패    : {stats['max_loss_streak']}회")

    current = stats["current_streak"]
    if current > 0:
        lines.append(f"  현재 상태    : {current}연승 중 ✨")
    elif current < 0:
        lines.append(f"  현재 상태    : {abs(current)}연패 중 ⚠️")

    # 연패 경고 (문서: 3연패→50%축소, 5연패→매매중단)
    if stats["max_loss_streak"] >= 3:
        lines.append(f"\n  ⚠️ 연속 {stats['max_loss_streak']}패 발생!")
        if stats["max_loss_streak"] >= 5:
            lines.append(f"     → 매매 일시 중단 권고 (문서: 5연패 시 중단)")
        else:
            lines.append(f"     → 포지션 크기 50% 축소 권고 (문서: 3연패 규칙)")

    # ── 전략별 분석 ──
    by_strat = strategy_breakdown(trades)
    if len(by_strat) > 1 or (len(by_strat) == 1 and not strategy):
        lines.append(f"\n  ■ 전략별 성과")
        lines.append(f"  {'─' * 50}")
        for s, st in sorted(by_strat.items()):
            target = TARGETS.get(s, {})
            wr_target = target.get("win_rate", (0, 100))
            rr_target = target.get("rr", 0)

            wr_ok = "✅" if wr_target[0] <= st["win_rate"] <= wr_target[1] else "❌"
            rr_ok = "✅" if st["risk_reward"] >= rr_target else "❌"

            lines.append(
                f"  {s:6s} │ {st['count']:3d}건 "
                f"│ 승률 {st['win_rate']:5.1f}% {wr_ok} (목표 {wr_target[0]}~{wr_target[1]}%) "
                f"│ 손익비 {st['risk_reward']:.2f} {rr_ok} (목표 {rr_target}+)"
            )

    # ── 감정 분석 ──
    emo = emotion_analysis(trades)
    if emo.get("entry_emotions"):
        lines.append(f"\n  ■ 감정 패턴 분석")
        lines.append(f"  {'─' * 50}")
        lines.append(f"  진입 시 감정 분포:")
        for em, cnt in emo["entry_emotions"].items():
            avg_pnl = emo["emotion_avg_pnl"].get(em, 0)
            icon = "📈" if avg_pnl > 0 else "📉"
            lines.append(f"    {em:6s} : {cnt}회  평균수익 {avg_pnl:+.2f}% {icon}")

    # ── 실수 패턴 ──
    mistakes = mistake_analysis(trades)
    if mistakes:
        issue_counter = Counter()
        for m in mistakes:
            for iss in m["issues"]:
                issue_counter[iss] += 1

        lines.append(f"\n  ■ 실수 패턴 (총 {len(mistakes)}건)")
        lines.append(f"  {'─' * 50}")
        for iss, cnt in issue_counter.most_common(5):
            lines.append(f"    {iss:24s} : {cnt}회")

    # ── 개선 포인트 ──
    lines.append(f"\n  ■ 개선 포인트")
    lines.append(f"  {'─' * 50}")
    improvements = _generate_improvements(stats, by_strat, emo, mistakes)
    for i, imp in enumerate(improvements, 1):
        lines.append(f"  {i}. {imp}")

    # ── 일일 메모 ──
    notes = db.get_daily_notes(start=start, end=end)
    if notes:
        lines.append(f"\n  ■ 일일 메모 (최근 {min(5, len(notes))}건)")
        lines.append(f"  {'─' * 50}")
        for n in notes[:5]:
            lines.append(f"  [{n['date']}] {n.get('lesson', '') or '-'}")

    lines.append(f"\n{'═' * 55}\n")
    return "\n".join(lines)


def _generate_improvements(
    stats: dict,
    by_strat: dict,
    emo: dict,
    mistakes: list,
) -> list[str]:
    """자동 개선 포인트 도출"""
    tips = []

    # 승률 기반
    if stats.get("win_rate", 0) < 40:
        tips.append("승률이 40% 미만입니다. 진입 조건을 더 엄격히 설정하세요.")

    # 손익비 기반
    if stats.get("risk_reward", 0) < 1.5:
        tips.append("손익비가 1.5 미만입니다. 손절을 더 타이트하게, 이익실현을 더 느리게 하세요.")

    # 기대값
    if stats.get("expectancy", 0) < 0:
        tips.append("기대값이 음수입니다. 현재 전략을 계속 사용하면 장기적으로 손실입니다.")

    # 계획 준수
    if stats.get("plan_follow_rate", 100) < 80:
        tips.append(
            f"계획 준수율이 {stats['plan_follow_rate']:.0f}%입니다. "
            "매매 전 계획서를 반드시 작성하세요."
        )

    # 감정 기반
    avg_pnl = emo.get("emotion_avg_pnl", {})
    for em in ["조급", "불안", "탐욕"]:
        if em in avg_pnl and avg_pnl[em] < -1:
            tips.append(f"'{em}' 상태에서 진입 시 평균 {avg_pnl[em]:.1f}% 손실. 해당 감정일 때 매매를 자제하세요.")

    # 전략별 목표 대비
    for s, st in by_strat.items():
        target = TARGETS.get(s, {})
        wr_target = target.get("win_rate", (0, 100))
        if st["win_rate"] < wr_target[0]:
            tips.append(f"[{s}] 승률 {st['win_rate']:.0f}% < 목표 {wr_target[0]}%. 진입 필터 강화 필요.")
        if st["risk_reward"] < target.get("rr", 0):
            tips.append(f"[{s}] 손익비 {st['risk_reward']:.1f} < 목표 {target['rr']}. 손절/목표가 재조정 필요.")

    # 실수 패턴 기반
    if mistakes:
        issue_counter = Counter()
        for m in mistakes:
            for iss in m["issues"]:
                issue_counter[iss] += 1
        top_issue = issue_counter.most_common(1)[0]
        tips.append(f"가장 빈번한 실수: '{top_issue[0]}' ({top_issue[1]}회). 체크리스트에 추가하세요.")

    if not tips:
        tips.append("현재 성과가 양호합니다. 현 전략을 유지하세요.")

    return tips[:7]
