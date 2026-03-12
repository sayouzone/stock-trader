"""
포지션 트레이딩 스크리너
- 200MA 위 시장, Stage 2, ADX>25, MA 풀백/돌파
- 승률 40~50%, 손익비 3:1+
"""
import pandas as pd
from scoring import ScreenResult, CheckItem


class PositionScreener:
    name = "포지션 트레이딩"

    @staticmethod
    def screen(ticker: str, df: pd.DataFrame) -> ScreenResult:
        r = ScreenResult(ticker=ticker, strategy="포지션 트레이딩", current_price=df["Close"].iloc[-1])
        last = df.iloc[-1]

        # 1. 200MA 위 (시장 환경)
        sma200 = last.get("SMA_200")
        above200 = not pd.isna(sma200) and last["Close"] > sma200
        r.checks.append(CheckItem("200MA 위 (강세장)", above200, f"{'✓' if above200 else '✗'}", 2.0))

        # 2. ADX > 25 (추세 강도)
        adx_val = last.get("ADX", 0)
        adx_ok = not pd.isna(adx_val) and adx_val > 25
        r.checks.append(CheckItem("ADX > 25", adx_ok, f"{adx_val:.1f}" if not pd.isna(adx_val) else "-", 1.5))

        # 3. Stage 2 (50MA 위 + 50MA 상승)
        sma50 = last.get("SMA_50")
        stage2 = False
        if not pd.isna(sma50) and len(df) > 10:
            prev_sma50 = df["SMA_50"].iloc[-10]
            stage2 = last["Close"] > sma50 and (pd.isna(prev_sma50) or sma50 > prev_sma50)
        r.checks.append(CheckItem("Stage 2 (50MA↑)", stage2, f"{'✓' if stage2 else '✗'}", 2.0))

        # 4. 52주 신고가 근접 (85%+)
        near_high = False
        h52 = last.get("High_52w")
        if not pd.isna(h52) and h52 > 0:
            pct = last["Close"] / h52
            near_high = pct >= 0.85
            r.checks.append(CheckItem("52주 고점 근접", near_high, f"{pct:.0%}", 1.0))
        else:
            r.checks.append(CheckItem("52주 고점 근접", False, "-", 1.0))

        # 5. MACD 양전환
        macd_h = last.get("MACD_Hist")
        prev_macd = df["MACD_Hist"].iloc[-2] if len(df) > 1 else None
        macd_cross = False
        if macd_h is not None and prev_macd is not None and not pd.isna(macd_h) and not pd.isna(prev_macd):
            macd_cross = prev_macd < 0 and macd_h > 0
        r.checks.append(CheckItem("MACD 시그널 교차", macd_cross, f"{'↑교차' if macd_cross else '-'}", 1.5))

        # 6. 거래량 증가
        vol_r = last.get("Vol_Ratio", 1)
        vol_ok = not pd.isna(vol_r) and vol_r > 1.5
        r.checks.append(CheckItem("거래량 150%+", vol_ok, f"{vol_r:.1f}x" if not pd.isna(vol_r) else "-", 1.0))

        # 진입 시그널 판정
        if above200 and stage2 and adx_ok:
            if macd_cross:
                r.entry_signals.append("MACD 시그널 상향 교차")
            sma20 = last.get("SMA_20")
            if sma20 and not pd.isna(sma20) and len(df) > 1:
                if df["Close"].iloc[-2] <= sma20 * 1.02 and last["Close"] > sma20:
                    r.entry_signals.append("20MA 풀백 반등")
            if vol_ok and near_high:
                r.entry_signals.append("Breakout 후보 (거래량+신고가)")

        # 손절가 (ATR × 2.5)
        atr_val = last.get("ATR", 0)
        if not pd.isna(atr_val) and atr_val > 0:
            r.stop_loss = round(last["Close"] - 2.5 * atr_val)

        return r
