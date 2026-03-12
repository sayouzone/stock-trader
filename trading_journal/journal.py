#!/usr/bin/env python3
"""
매매일지 & 복기 시스템 - CLI 인터페이스
=============================================
주식 투자 기법 종합 가이드에 기반한 매매 기록 + 복기 도구

사용법:
    python journal.py add                  # 매매 기록 추가
    python journal.py close <ID>           # 매매 청산
    python journal.py list                 # 거래 목록
    python journal.py open                 # 미청산 포지션
    python journal.py report               # 전체 복기 리포트
    python journal.py report --month 2025-03  # 월별 복기
    python journal.py report --strategy 스윙  # 전략별 복기
    python journal.py stats                # 빠른 통계
    python journal.py note                 # 일일 메모
    python journal.py demo                 # 데모 데이터 생성
"""

import argparse
import random
import sys
from datetime import date, datetime, timedelta

from tabulate import tabulate

import db
import analytics


# ── 헬퍼 ──

def _prompt(msg: str, default: str = "", choices: list = None) -> str:
    """사용자 입력 프롬프트"""
    suffix = ""
    if default:
        suffix = f" [{default}]"
    if choices:
        numbered = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(choices))
        print(f"\n{numbered}")
        suffix = f" [1-{len(choices)}]"

    val = input(f"  ? {msg}{suffix}: ").strip()

    if choices:
        try:
            idx = int(val) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        except (ValueError, IndexError):
            pass
        return val if val else default

    return val if val else default


def _format_pnl(val: float) -> str:
    if val > 0:
        return f"\033[32m+{val:.2f}%\033[0m"
    elif val < 0:
        return f"\033[31m{val:.2f}%\033[0m"
    return f"{val:.2f}%"


# ── 명령어 ──

def cmd_add(args):
    """매매 기록 추가"""
    print(f"\n{'─' * 45}")
    print("  📝 매매 기록 추가")
    print(f"{'─' * 45}")

    ticker = _prompt("종목코드", "").upper()
    if not ticker:
        print("  종목코드를 입력하세요.")
        return

    strategy = _prompt("전략", choices=db.STRATEGIES)
    entry_date = _prompt("진입일 (YYYY-MM-DD)", date.today().isoformat())
    entry_price = float(_prompt("진입가", "0"))
    entry_shares = int(_prompt("수량", "0"))

    print("\n  [진입 사유]")
    entry_reason = _prompt("사유", choices=db.ENTRY_REASONS)
    entry_note = _prompt("메모 (자유기입)", "")

    stop_loss = _prompt("손절가", "")
    stop_loss = float(stop_loss) if stop_loss else None

    target_price = _prompt("목표가", "")
    target_price = float(target_price) if target_price else None

    risk_pct = None
    if stop_loss and entry_price > 0:
        risk_per_share = abs(entry_price - stop_loss)
        risk_pct_auto = risk_per_share / entry_price * 100
        print(f"  → 자동 계산 리스크: {risk_pct_auto:.2f}%")
        risk_pct = risk_pct_auto

    print("\n  [심리 기록]")
    entry_emotion = _prompt("진입 시 감정", choices=db.EMOTIONS)

    tags = _prompt("태그 (쉼표 구분)", "")

    trade_id = db.add_trade(
        ticker=ticker,
        strategy=strategy,
        entry_date=entry_date,
        entry_price=entry_price,
        entry_shares=entry_shares,
        entry_reason=entry_reason,
        entry_note=entry_note,
        stop_loss=stop_loss,
        target_price=target_price,
        risk_pct=risk_pct,
        entry_emotion=entry_emotion,
        tags=tags,
    )

    print(f"\n  ✅ 거래 #{trade_id} 기록 완료")
    print(f"     {ticker} {entry_shares}주 @ {entry_price:,.0f}")
    if stop_loss:
        print(f"     손절가: {stop_loss:,.0f}")
    if target_price:
        print(f"     목표가: {target_price:,.0f}")


def cmd_close(args):
    """매매 청산"""
    trade_id = args.id

    # 미청산 거래 확인
    open_trades = db.get_open_trades()
    trade = None
    for t in open_trades:
        if t["id"] == trade_id:
            trade = t
            break

    if not trade:
        print(f"  ❌ 미청산 거래 #{trade_id}를 찾을 수 없습니다.")
        if open_trades:
            print(f"  미청산 거래 ID: {[t['id'] for t in open_trades]}")
        return

    print(f"\n{'─' * 45}")
    print(f"  📤 매매 청산 - #{trade_id} {trade['ticker']}")
    print(f"     진입: {trade['entry_date']} @ {trade['entry_price']:,.0f} × {trade['entry_shares']}주")
    print(f"{'─' * 45}")

    exit_date = _prompt("청산일 (YYYY-MM-DD)", date.today().isoformat())
    exit_price = float(_prompt("청산가", "0"))
    exit_shares = _prompt("청산 수량 (전량 Enter)", "")
    exit_shares = int(exit_shares) if exit_shares else None

    print("\n  [청산 사유]")
    exit_reason = _prompt("사유", choices=db.EXIT_REASONS)
    exit_note = _prompt("메모 (자유기입)", "")

    print("\n  [심리 기록]")
    exit_emotion = _prompt("청산 시 감정", choices=db.EMOTIONS)

    plan_yn = _prompt("매매 계획을 준수했나요? (Y/n)", "Y")
    plan_followed = plan_yn.upper() != "N"

    result = db.close_trade(
        trade_id=trade_id,
        exit_date=exit_date,
        exit_price=exit_price,
        exit_shares=exit_shares,
        exit_reason=exit_reason,
        exit_note=exit_note,
        exit_emotion=exit_emotion,
        plan_followed=plan_followed,
    )

    pnl_str = _format_pnl(result["pnl_pct"])
    r_str = f"{result['r_multiple']:.2f}R" if result.get("r_multiple") else "-"

    print(f"\n  ✅ 청산 완료")
    print(f"     {result['ticker']} │ 수익률: {pnl_str} │ R: {r_str} │ {result['holding_days']}일 보유")
    print(f"     손익: {result['pnl']:+,.0f}원")


def cmd_list(args):
    """거래 목록"""
    limit = args.limit or 20
    trades = db.get_closed_trades(
        strategy=args.strategy,
        limit=limit,
    )

    if not trades:
        print("  기록된 거래가 없습니다.")
        return

    rows = []
    for t in trades:
        pnl_pct = t.get("pnl_pct", 0) or 0
        r_mul = t.get("r_multiple")
        rows.append({
            "ID": t["id"],
            "종목": t["ticker"],
            "전략": t["strategy"][:3],
            "진입일": t["entry_date"],
            "청산일": t["exit_date"],
            "수익률": f"{pnl_pct:+.2f}%",
            "R배수": f"{r_mul:.2f}" if r_mul else "-",
            "보유일": t.get("holding_days", "-"),
            "사유": (t.get("exit_reason") or "")[:12],
        })

    print(f"\n  최근 {len(rows)}건 거래:")
    print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))


def cmd_open(args):
    """미청산 포지션"""
    trades = db.get_open_trades()

    if not trades:
        print("  미청산 포지션이 없습니다.")
        return

    rows = []
    for t in trades:
        hold = (date.today() - datetime.strptime(t["entry_date"], "%Y-%m-%d").date()).days
        rows.append({
            "ID": t["id"],
            "종목": t["ticker"],
            "전략": t["strategy"][:3],
            "진입일": t["entry_date"],
            "진입가": f"{t['entry_price']:,.0f}",
            "수량": t["entry_shares"],
            "손절가": f"{t['stop_loss']:,.0f}" if t.get("stop_loss") else "-",
            "목표가": f"{t['target_price']:,.0f}" if t.get("target_price") else "-",
            "보유일": hold,
        })

    print(f"\n  📊 미청산 포지션 ({len(rows)}건):")
    print(tabulate(rows, headers="keys", tablefmt="simple", stralign="right"))


def cmd_report(args):
    """복기 리포트"""
    start = end = None
    strategy = args.strategy

    if args.month:
        # 월별: "2025-03" 형식
        year, month = args.month.split("-")
        start = f"{year}-{month}-01"
        if int(month) == 12:
            end = f"{int(year)+1}-01-01"
        else:
            end = f"{year}-{int(month)+1:02d}-01"
    elif args.start:
        start = args.start
        end = args.end

    report = analytics.generate_report(start=start, end=end, strategy=strategy)
    print(report)


def cmd_stats(args):
    """빠른 통계 요약"""
    all_trades = db.get_all_trades(limit=9999)
    stats = analytics.compute_stats(all_trades)

    if stats.get("count", 0) == 0:
        print("  거래 기록이 없습니다.")
        return

    print(f"\n  📊 전체 통계 요약")
    print(f"  {'─' * 40}")
    print(f"  거래: {stats['count']}건 │ 미청산: {stats['open_count']}건")
    print(f"  승률: {stats['win_rate']:.1f}% │ 손익비: {stats['risk_reward']:.2f}")
    print(f"  기대값: {stats['expectancy']:+.2f}% │ 평균R: {stats['avg_r']:.2f}")
    print(f"  총 손익: {stats['total_pnl']:+,.0f}원")
    print(f"  연승: {stats['max_win_streak']} │ 연패: {stats['max_loss_streak']}")
    print(f"  계획준수: {stats['plan_follow_rate']:.0f}%")


def cmd_note(args):
    """일일 메모"""
    print(f"\n{'─' * 45}")
    print(f"  📓 일일 메모")
    print(f"{'─' * 45}")

    dt = _prompt("날짜 (YYYY-MM-DD)", date.today().isoformat())
    market = _prompt("시장 환경 메모", "")
    lesson = _prompt("오늘의 교훈", "")
    mood = _prompt("심리 상태", choices=db.EMOTIONS)

    db.add_daily_note(dt, market, lesson, mood)
    print(f"  ✅ {dt} 메모 저장 완료")


def cmd_demo(args):
    """데모 데이터 생성"""
    print(f"\n  🎲 데모 데이터 생성 중...")

    strategies = db.STRATEGIES
    tickers_kr = [
        "삼성전자", "SK하이닉스", "NAVER", "카카오", "현대차",
        "LG에너지", "셀트리온", "KB금융", "삼성SDI", "기아",
    ]
    tickers_us = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
        "META", "TSLA", "JPM", "JNJ", "BRK-B",
    ]
    tickers = tickers_kr + tickers_us

    base = datetime(2024, 6, 1)
    count = args.count or 40

    for i in range(count):
        ticker = random.choice(tickers)
        strategy = random.choice(strategies)
        entry_date = base + timedelta(days=random.randint(0, 250))
        entry_price = random.uniform(10000, 500000) if ticker in tickers_kr else random.uniform(50, 1000)
        entry_shares = random.randint(5, 200)

        stop_pct = random.uniform(0.03, 0.10)
        stop_loss = entry_price * (1 - stop_pct)
        target_price = entry_price * (1 + stop_pct * random.uniform(2, 4))

        entry_reason = random.choice(db.ENTRY_REASONS)
        entry_emotion = random.choice(db.EMOTIONS)

        trade_id = db.add_trade(
            ticker=ticker,
            strategy=strategy,
            entry_date=entry_date.strftime("%Y-%m-%d"),
            entry_price=round(entry_price, 0),
            entry_shares=entry_shares,
            entry_reason=entry_reason,
            stop_loss=round(stop_loss, 0),
            target_price=round(target_price, 0),
            risk_pct=stop_pct * 100,
            entry_emotion=entry_emotion,
        )

        # 80% 확률로 청산
        if random.random() < 0.80:
            hold_days = random.randint(1, 90)
            exit_date = entry_date + timedelta(days=hold_days)
            # 랜덤 수익률 (약간 양의 편향)
            ret = random.gauss(0.005, 0.08)
            exit_price = entry_price * (1 + ret)

            exit_reason = random.choice(db.EXIT_REASONS)
            exit_emotion = random.choice(db.EMOTIONS)
            plan = random.random() < 0.75

            db.close_trade(
                trade_id=trade_id,
                exit_date=exit_date.strftime("%Y-%m-%d"),
                exit_price=round(exit_price, 0),
                exit_reason=exit_reason,
                exit_emotion=exit_emotion,
                plan_followed=plan,
            )

    # 일일 메모 몇 개
    for i in range(10):
        dt = base + timedelta(days=random.randint(0, 250))
        lessons = [
            "추격매수 금지 - 피벗에서만 진입하자",
            "손절은 기계적으로 실행해야 한다",
            "시장 방향을 먼저 확인하고 종목을 봐야 한다",
            "감정적일 때는 매매를 쉬어야 한다",
            "거래량 확인 없이 들어가면 안 된다",
            "분할 매수 원칙을 지키자",
            "이익을 너무 일찍 실현하지 말자",
        ]
        db.add_daily_note(
            dt.strftime("%Y-%m-%d"),
            market_condition=random.choice(["강세", "약세", "횡보", "변동성 확대"]),
            lesson=random.choice(lessons),
            mood=random.choice(db.EMOTIONS),
        )

    print(f"  ✅ {count}건의 매매 + 10건의 일일메모 생성 완료")
    print(f"  → 'python journal.py report'로 복기 리포트를 확인하세요")


# ── 메인 ──

def main():
    db.init_db()

    parser = argparse.ArgumentParser(
        description="매매일지 & 복기 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python journal.py add                    매매 기록 추가
  python journal.py close 3               #3 거래 청산
  python journal.py list                   최근 거래 목록
  python journal.py open                   미청산 포지션
  python journal.py report                 전체 복기 리포트
  python journal.py report --month 2025-03 월별 복기
  python journal.py report --strategy 스윙 전략별 복기
  python journal.py stats                  빠른 통계
  python journal.py note                   일일 메모
  python journal.py demo                   데모 데이터 생성
        """,
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("add", help="매매 기록 추가")

    close_p = sub.add_parser("close", help="매매 청산")
    close_p.add_argument("id", type=int, help="거래 ID")

    list_p = sub.add_parser("list", help="거래 목록")
    list_p.add_argument("--limit", "-n", type=int, default=20)
    list_p.add_argument("--strategy", "-s", type=str)

    sub.add_parser("open", help="미청산 포지션")

    report_p = sub.add_parser("report", help="복기 리포트")
    report_p.add_argument("--month", "-m", type=str, help="YYYY-MM")
    report_p.add_argument("--start", type=str)
    report_p.add_argument("--end", type=str)
    report_p.add_argument("--strategy", "-s", type=str)

    sub.add_parser("stats", help="빠른 통계")
    sub.add_parser("note", help="일일 메모")

    demo_p = sub.add_parser("demo", help="데모 데이터 생성")
    demo_p.add_argument("--count", "-c", type=int, default=40)

    args = parser.parse_args()

    commands = {
        "add": cmd_add,
        "close": cmd_close,
        "list": cmd_list,
        "open": cmd_open,
        "report": cmd_report,
        "stats": cmd_stats,
        "note": cmd_note,
        "demo": cmd_demo,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
