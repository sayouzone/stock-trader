"""
SQLite 데이터베이스 모듈
- 매매일지 스키마 정의 및 CRUD
- 문서의 매매 프로세스 구조를 반영한 컬럼 설계
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "journal.db"

# ── 문서 기반 상수 ──

STRATEGIES = [
    "포지션", "성장주", "가치", "스윙", "모멘텀", "배당",
]

EMOTIONS = [
    "차분함", "확신", "불안", "조급", "탐욕", "공포", "흥분", "후회", "무감정",
]

ENTRY_REASONS = [
    "MA풀백반등", "돌파(Breakout)", "MACD교차", "볼린저Squeeze",
    "피벗돌파", "RSI과매도반전", "골든크로스", "MACD_0선돌파",
    "모멘텀상위편입", "배당밴드상단", "스토캐스틱교차",
    "캔들패턴", "피보나치되돌림", "기타",
]

EXIT_REASONS = [
    "손절(ATR)", "손절(고정%)", "손절(Thesis)", "손절(시간)",
    "MA이탈", "다이버전스", "ADX하락", "클라이맥스탑",
    "목표가도달", "이익실현(분할)", "리밸런싱",
    "내재가치도달", "과대평가", "배당삭감", "어닝쇼크", "기타",
]


def get_db_path() -> Path:
    return DB_PATH


@contextmanager
def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """데이터베이스 초기화 (테이블 생성)"""
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT NOT NULL,
            strategy        TEXT NOT NULL,
            direction       TEXT DEFAULT 'LONG' CHECK(direction IN ('LONG','SHORT')),

            -- 진입
            entry_date      TEXT NOT NULL,
            entry_price     REAL NOT NULL,
            entry_shares    INTEGER NOT NULL,
            entry_reason    TEXT,
            entry_note      TEXT,

            -- 청산 (NULL이면 미청산)
            exit_date       TEXT,
            exit_price      REAL,
            exit_shares     INTEGER,
            exit_reason     TEXT,
            exit_note       TEXT,

            -- 리스크 관리
            stop_loss       REAL,
            target_price    REAL,
            risk_pct        REAL,           -- 이 거래의 계좌 리스크 %

            -- 심리 기록
            entry_emotion   TEXT,
            exit_emotion    TEXT,
            plan_followed   INTEGER DEFAULT 1, -- 매매 계획 준수 여부 (0/1)

            -- 자동 계산 (청산 시 업데이트)
            pnl             REAL,
            pnl_pct         REAL,
            r_multiple      REAL,           -- 수익/리스크
            holding_days    INTEGER,

            -- 메타
            created_at      TEXT DEFAULT (datetime('now','localtime')),
            tags            TEXT            -- 쉼표 구분 태그
        );

        CREATE TABLE IF NOT EXISTS daily_notes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT NOT NULL UNIQUE,
            market_condition TEXT,           -- 시장 환경 메모
            lesson          TEXT,           -- 오늘의 교훈
            mood            TEXT,           -- 심리 상태
            created_at      TEXT DEFAULT (datetime('now','localtime'))
        );

        CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
        CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
        CREATE INDEX IF NOT EXISTS idx_trades_entry_date ON trades(entry_date);
        CREATE INDEX IF NOT EXISTS idx_trades_exit_date ON trades(exit_date);
        """)


# ── CRUD ──

def add_trade(
    ticker: str,
    strategy: str,
    entry_date: str,
    entry_price: float,
    entry_shares: int,
    entry_reason: str = "",
    entry_note: str = "",
    stop_loss: float = None,
    target_price: float = None,
    risk_pct: float = None,
    entry_emotion: str = "",
    tags: str = "",
) -> int:
    """매매 기록 추가 (진입)"""
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO trades
            (ticker, strategy, entry_date, entry_price, entry_shares,
             entry_reason, entry_note, stop_loss, target_price, risk_pct,
             entry_emotion, tags)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ticker.upper(), strategy, entry_date, entry_price, entry_shares,
             entry_reason, entry_note, stop_loss, target_price, risk_pct,
             entry_emotion, tags),
        )
        return cur.lastrowid


def close_trade(
    trade_id: int,
    exit_date: str,
    exit_price: float,
    exit_shares: int = None,
    exit_reason: str = "",
    exit_note: str = "",
    exit_emotion: str = "",
    plan_followed: bool = True,
) -> dict:
    """매매 청산 기록"""
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
        if not row:
            raise ValueError(f"Trade #{trade_id} not found")

        if exit_shares is None:
            exit_shares = row["entry_shares"]

        # 수익 계산
        pnl = (exit_price - row["entry_price"]) * exit_shares
        pnl_pct = (exit_price - row["entry_price"]) / row["entry_price"] * 100

        # R 배수 계산
        r_multiple = None
        if row["stop_loss"] and row["stop_loss"] > 0:
            risk_per_share = abs(row["entry_price"] - row["stop_loss"])
            if risk_per_share > 0:
                r_multiple = (exit_price - row["entry_price"]) / risk_per_share

        # 보유일
        entry_dt = datetime.strptime(row["entry_date"], "%Y-%m-%d")
        exit_dt = datetime.strptime(exit_date, "%Y-%m-%d")
        holding_days = (exit_dt - entry_dt).days

        conn.execute(
            """UPDATE trades SET
            exit_date=?, exit_price=?, exit_shares=?, exit_reason=?,
            exit_note=?, exit_emotion=?, plan_followed=?,
            pnl=?, pnl_pct=?, r_multiple=?, holding_days=?
            WHERE id=?""",
            (exit_date, exit_price, exit_shares, exit_reason,
             exit_note, exit_emotion, 1 if plan_followed else 0,
             pnl, pnl_pct, r_multiple, holding_days, trade_id),
        )

        return {
            "id": trade_id,
            "ticker": row["ticker"],
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "r_multiple": r_multiple,
            "holding_days": holding_days,
        }


def get_open_trades() -> list[dict]:
    """미청산 포지션 조회"""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE exit_date IS NULL ORDER BY entry_date DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_closed_trades(
    start: str = None,
    end: str = None,
    strategy: str = None,
    ticker: str = None,
    limit: int = 100,
) -> list[dict]:
    """청산된 거래 조회"""
    query = "SELECT * FROM trades WHERE exit_date IS NOT NULL"
    params = []

    if start:
        query += " AND exit_date >= ?"
        params.append(start)
    if end:
        query += " AND exit_date <= ?"
        params.append(end)
    if strategy:
        query += " AND strategy = ?"
        params.append(strategy)
    if ticker:
        query += " AND ticker = ?"
        params.append(ticker.upper())

    query += " ORDER BY exit_date DESC LIMIT ?"
    params.append(limit)

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def get_all_trades(limit: int = 200) -> list[dict]:
    """전체 거래 조회"""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY entry_date DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def add_daily_note(
    dt: str,
    market_condition: str = "",
    lesson: str = "",
    mood: str = "",
) -> int:
    """일일 메모 추가"""
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT OR REPLACE INTO daily_notes
            (date, market_condition, lesson, mood)
            VALUES (?,?,?,?)""",
            (dt, market_condition, lesson, mood),
        )
        return cur.lastrowid


def get_daily_notes(start: str = None, end: str = None) -> list[dict]:
    """일일 메모 조회"""
    query = "SELECT * FROM daily_notes WHERE 1=1"
    params = []
    if start:
        query += " AND date >= ?"
        params.append(start)
    if end:
        query += " AND date <= ?"
        params.append(end)
    query += " ORDER BY date DESC"

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
