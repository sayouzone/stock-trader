"""
스윙 트레이딩 스크리너
- RSI 과매도 반전, MA 풀백, 스토캐스틱 교차
- 승률 50~60%, 손익비 1.5:1~2:1, 시간 정지 8일
"""
import pandas as pd
from scoring import ScreenResult, CheckItem


class SwingScreener:
    name = "스윙 트레이딩"

    @staticmethod
    def screen(ticker: str, df: pd.DataFrame) -> ScreenResult:
        r = ScreenResult(ticker=ticker, strategy="스윙 트레이딩", current_price=df["Close"].iloc[-1])
        last = df.iloc[-1]

        # 1. 20MA 위 (기본 추세 확인)
        sma20 = last.get("SMA_20")
        above20 = not pd.isna(sma20) and last["Close"] > sma20
        r.checks.append(CheckItem("20MA 위 (단기 추세)", above20,
                                   f"{'✓' if above20 else '✗'}", 1.5))

        # 2. RSI 과매도 반전 (30 부근에서 상승)
        rsi_val = last.get("RSI")
        rsi_reversal = False
        if not pd.isna(rsi_val) and len(df) > 3:
            rsi_prev = df["RSI"].iloc[-3:-1]
            had_oversold = any(not pd.isna(v) and v < 35 for v in rsi_prev)
            rsi_reversal = had_oversold and rsi_val > 35
        r.checks.append(CheckItem("RSI 과매도 반전", rsi_reversal,
                                   f"{rsi_val:.0f}" if not pd.isna(rsi_val) else "-", 2.0))

        # 3. 스토캐스틱 골든크로스 (%K > %D, 둘 다 < 30에서 상승)
        stoch_k = last.get("Stoch_K")
        stoch_d = last.get("Stoch_D")
        stoch_cross = False
        if not pd.isna(stoch_k) and not pd.isna(stoch_d) and len(df) > 1:
            prev_k = df["Stoch_K"].iloc[-2]
            prev_d = df["Stoch_D"].iloc[-2]
            if (not pd.isna(prev_k) and not pd.isna(prev_d)):
                stoch_cross = prev_k <= prev_d and stoch_k > stoch_d and stoch_k < 50
        r.checks.append(CheckItem("스토캐스틱 교차", stoch_cross,
                                   f"K={stoch_k:.0f}" if not pd.isna(stoch_k) else "-", 2.0))

        # 4. 볼린저 밴드 하단 터치 반등
        bb_lower = last.get("BB_Lower")
        bb_bounce = False
        if not pd.isna(bb_lower) and len(df) > 3:
            recent_low = df["Low"].iloc[-3:].min()
            bb_bounce = recent_low <= bb_lower * 1.01 and last["Close"] > bb_lower
        r.checks.append(CheckItem("BB 하단 반등", bb_bounce,
                                   f"{'✓' if bb_bounce else '-'}", 1.5))

        # 5. ATR 적정 범위 (변동성 너무 크지 않음)
        atr_val = last.get("ATR", 0)
        atr_pct = atr_val / last["Close"] * 100 if last["Close"] > 0 and not pd.isna(atr_val) else 0
        atr_ok = 1.0 <= atr_pct <= 5.0
        r.checks.append(CheckItem("ATR 적정 (1~5%)", atr_ok,
                                   f"{atr_pct:.1f}%", 1.0))

        # 6. 거래량 증가
        vol_r = last.get("Vol_Ratio", 1)
        vol_ok = not pd.isna(vol_r) and vol_r > 1.3
        r.checks.append(CheckItem("거래량 130%+", vol_ok,
                                   f"{vol_r:.1f}x" if not pd.isna(vol_r) else "-", 1.0))

        # 진입 시그널
        if rsi_reversal:
            r.entry_signals.append("RSI 과매도 반전 매수")
        if stoch_cross and bb_bounce:
            r.entry_signals.append("BB 하단 + 스토캐스틱 교차")
        if above20 and len(df) > 2:
            # MA 풀백 패턴
            if df["Low"].iloc[-2] <= sma20 * 1.01 and last["Close"] > sma20:
                r.entry_signals.append("20MA 풀백 반등")

        # 손절가 (ATR × 1.5)
        if not pd.isna(atr_val) and atr_val > 0:
            r.stop_loss = round(last["Close"] - 1.5 * atr_val)
            # 목표가 (2R)
            risk = last["Close"] - r.stop_loss
            r.target_price = round(last["Close"] + 2.0 * risk)

        return r
