"""
배당 투자 스크리너
- 저변동성, 52주 저점 근접 (수익률 밴드 하단)
- DCA (분할 매수) 전략, 안정적 수익 추구
"""
import pandas as pd
from scoring import ScreenResult, CheckItem


class DividendScreener:
    name = "배당 투자"

    @staticmethod
    def screen(ticker: str, df: pd.DataFrame) -> ScreenResult:
        r = ScreenResult(ticker=ticker, strategy="배당 투자", current_price=df["Close"].iloc[-1])
        last = df.iloc[-1]

        # 1. 저변동성 (ATR/가격 < 2%)
        atr_val = last.get("ATR", 0)
        atr_pct = atr_val / last["Close"] * 100 if last["Close"] > 0 and not pd.isna(atr_val) else 999
        low_vol = atr_pct < 2.0
        r.checks.append(CheckItem("저변동성 (ATR<2%)", low_vol,
                                   f"{atr_pct:.1f}%", 2.0))

        # 2. 52주 저점 근접 (수익률 밴드 하단 = 가격 하단)
        near_low = False
        l52 = last.get("Low_52w")
        h52 = last.get("High_52w")
        if not pd.isna(l52) and not pd.isna(h52) and h52 > l52:
            price_band = (last["Close"] - l52) / (h52 - l52)
            near_low = price_band <= 0.35  # 하위 35%
            r.checks.append(CheckItem("가격밴드 하단 (35%↓)", near_low,
                                       f"{price_band:.0%}", 2.0))
        else:
            r.checks.append(CheckItem("가격밴드 하단", False, "-", 2.0))

        # 3. BB 하단 근접 (매수 타이밍)
        bb_lower = last.get("BB_Lower")
        bb_middle = last.get("BB_Middle")
        bb_near = False
        if not pd.isna(bb_lower) and not pd.isna(bb_middle):
            bb_pos = (last["Close"] - bb_lower) / (bb_middle - bb_lower) if bb_middle != bb_lower else 1
            bb_near = bb_pos < 0.5
            r.checks.append(CheckItem("BB 하단 근접", bb_near,
                                       f"{bb_pos:.0%}", 1.5))
        else:
            r.checks.append(CheckItem("BB 하단 근접", False, "-", 1.5))

        # 4. 장기 우상향 (200MA 상승)
        sma200 = last.get("SMA_200")
        ma200_up = False
        if not pd.isna(sma200) and len(df) > 63:
            prev200 = df["SMA_200"].iloc[-63]
            ma200_up = not pd.isna(prev200) and sma200 > prev200
        r.checks.append(CheckItem("200MA 3개월 상승", ma200_up,
                                   f"{'↑' if ma200_up else '→'}", 1.5))

        # 5. RSI 중립~과매도 (< 55)
        rsi_val = last.get("RSI")
        rsi_ok = not pd.isna(rsi_val) and rsi_val < 55
        r.checks.append(CheckItem("RSI < 55 (과매수X)", rsi_ok,
                                   f"{rsi_val:.0f}" if not pd.isna(rsi_val) else "-", 1.0))

        # 6. 최근 급락 없음 (1개월 -10% 이상 없음)
        ret_21 = last.get("Return_21d", 0)
        no_crash = not pd.isna(ret_21) and ret_21 > -0.10
        r.checks.append(CheckItem("1개월 낙폭 < 10%", no_crash,
                                   f"{ret_21:.1%}" if not pd.isna(ret_21) else "-", 1.0))

        # 진입 시그널
        if low_vol:
            if near_low and bb_near:
                r.entry_signals.append("수익률 밴드 하단 DCA 매수")
            if ma200_up and rsi_ok:
                r.entry_signals.append("장기 상승 + 저가 매수 기회")
            if no_crash and bb_near:
                r.entry_signals.append("안정적 BB 하단 분할 매수")

        # 손절가 (52주 저점 하회)
        if not pd.isna(l52):
            r.stop_loss = round(l52 * 0.97)  # 저점 3% 하회 시 손절

        return r
