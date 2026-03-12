"""
가치 투자 스크리너
- 52주 저점 근접, 골든크로스, MACD 0선 돌파
- 승률 55~65%, 손익비 2:1~3:1
"""
import pandas as pd
from scoring import ScreenResult, CheckItem


class ValueScreener:
    name = "가치 트레이딩"

    @staticmethod
    def screen(ticker: str, df: pd.DataFrame) -> ScreenResult:
        r = ScreenResult(ticker=ticker, strategy="가치 트레이딩", current_price=df["Close"].iloc[-1])
        last = df.iloc[-1]

        # 1. 52주 저점 근접 (저점 대비 +30% 이내)
        near_low = False
        l52 = last.get("Low_52w")
        if not pd.isna(l52) and l52 > 0:
            pct_from_low = (last["Close"] - l52) / l52
            near_low = pct_from_low <= 0.30
            r.checks.append(CheckItem("52주 저점 근접", near_low,
                                       f"+{pct_from_low:.0%}", 2.0))
        else:
            r.checks.append(CheckItem("52주 저점 근접", False, "-", 2.0))

        # 2. 골든크로스 (50MA > 200MA)
        sma50 = last.get("SMA_50")
        sma200 = last.get("SMA_200")
        golden = False
        if not pd.isna(sma50) and not pd.isna(sma200):
            golden = sma50 > sma200
        r.checks.append(CheckItem("골든크로스 (50>200)", golden,
                                   f"{'✓' if golden else '✗'}", 2.0))

        # 3. 골든크로스 전환 시점 (최근 20일 이내 교차)
        recent_cross = False
        if "SMA_50" in df.columns and "SMA_200" in df.columns and len(df) > 20:
            for i in range(-20, -1):
                s50_prev = df["SMA_50"].iloc[i - 1]
                s200_prev = df["SMA_200"].iloc[i - 1]
                s50_cur = df["SMA_50"].iloc[i]
                s200_cur = df["SMA_200"].iloc[i]
                if (not pd.isna(s50_prev) and not pd.isna(s200_prev)
                        and not pd.isna(s50_cur) and not pd.isna(s200_cur)):
                    if s50_prev <= s200_prev and s50_cur > s200_cur:
                        recent_cross = True
                        break
        r.checks.append(CheckItem("최근 골든크로스 발생", recent_cross,
                                   f"{'✓' if recent_cross else '-'}", 1.5))

        # 4. MACD 0선 근접/돌파
        macd_val = last.get("MACD")
        macd_near_zero = False
        if not pd.isna(macd_val):
            macd_near_zero = -0.5 <= macd_val or macd_val > 0
        r.checks.append(CheckItem("MACD ≥ 0선", macd_near_zero,
                                   f"{macd_val:.2f}" if not pd.isna(macd_val) else "-", 1.5))

        # 5. RSI 과매도 탈출 (30~50 영역)
        rsi_val = last.get("RSI")
        rsi_recovery = False
        if not pd.isna(rsi_val):
            rsi_recovery = 30 <= rsi_val <= 55
        r.checks.append(CheckItem("RSI 반등 영역", rsi_recovery,
                                   f"{rsi_val:.0f}" if not pd.isna(rsi_val) else "-", 1.0))

        # 6. 거래량 증가 추세
        vol_r = last.get("Vol_Ratio", 1)
        vol_ok = not pd.isna(vol_r) and vol_r > 1.2
        r.checks.append(CheckItem("거래량 120%+", vol_ok,
                                   f"{vol_r:.1f}x" if not pd.isna(vol_r) else "-", 1.0))

        # 진입 시그널
        if near_low:
            if recent_cross:
                r.entry_signals.append("골든크로스 전환 (매수 시점)")
            if golden and macd_near_zero:
                r.entry_signals.append("MACD 0선 돌파 (추세 전환)")
            if rsi_recovery and vol_ok:
                r.entry_signals.append("과매도 탈출 + 거래량 증가")

        # 손절가 (최근 저점 - ATR)
        atr_val = last.get("ATR", 0)
        if not pd.isna(atr_val) and atr_val > 0 and len(df) > 20:
            recent_low = df["Low"].iloc[-20:].min()
            r.stop_loss = round(recent_low - atr_val)

        # 목표가 (리레이팅: 52주 고점)
        h52 = last.get("High_52w")
        if not pd.isna(h52):
            r.target_price = round(h52)

        return r
