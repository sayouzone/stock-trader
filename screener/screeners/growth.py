"""
성장주 트레이딩 스크리너 (CAN SLIM / SEPA 기반)
- 피봇 돌파, 거래량 150%+, 50/150/200MA 정배열
- 승률 35~45%, 손익비 3:1~5:1
"""
import pandas as pd
from scoring import ScreenResult, CheckItem


class GrowthScreener:
    name = "성장주 트레이딩"

    @staticmethod
    def screen(ticker: str, df: pd.DataFrame) -> ScreenResult:
        r = ScreenResult(ticker=ticker, strategy="성장주 트레이딩", current_price=df["Close"].iloc[-1])
        last = df.iloc[-1]

        # 1. 이동평균 정배열 (50 > 150 > 200)
        sma50 = last.get("SMA_50")
        sma150 = last.get("SMA_150")
        sma200 = last.get("SMA_200")
        ma_aligned = False
        if all(not pd.isna(v) for v in [sma50, sma150, sma200]):
            ma_aligned = sma50 > sma150 > sma200
        r.checks.append(CheckItem("MA 정배열 (50>150>200)", ma_aligned,
                                   f"{'✓' if ma_aligned else '✗'}", 2.5))

        # 2. 200MA 상승 추세 (최근 1개월)
        sma200_rising = False
        if not pd.isna(sma200) and len(df) > 21:
            prev200 = df["SMA_200"].iloc[-21]
            sma200_rising = not pd.isna(prev200) and sma200 > prev200
        r.checks.append(CheckItem("200MA 상승 추세", sma200_rising,
                                   f"{'↑' if sma200_rising else '→'}", 1.5))

        # 3. 52주 신고가 근접 (90%+) — 피봇 영역
        near_high = False
        h52 = last.get("High_52w")
        if not pd.isna(h52) and h52 > 0:
            pct = last["Close"] / h52
            near_high = pct >= 0.90
            r.checks.append(CheckItem("52주 고점 90%+", near_high, f"{pct:.0%}", 2.0))
        else:
            r.checks.append(CheckItem("52주 고점 90%+", False, "-", 2.0))

        # 4. RS (상대 강도) — 6개월 수익률 상위
        ret_126 = last.get("Return_126d")
        rs_ok = not pd.isna(ret_126) and ret_126 > 0.20
        r.checks.append(CheckItem("6개월 수익률 > 20%", rs_ok,
                                   f"{ret_126:.0%}" if not pd.isna(ret_126) else "-", 1.5))

        # 5. 거래량 급등 (150%+)
        vol_r = last.get("Vol_Ratio", 1)
        vol_ok = not pd.isna(vol_r) and vol_r > 1.5
        r.checks.append(CheckItem("거래량 150%+", vol_ok,
                                   f"{vol_r:.1f}x" if not pd.isna(vol_r) else "-", 1.5))

        # 6. 볼린저 밴드 스퀴즈 해소 (BB Width 확대)
        bb_w = last.get("BB_Width")
        bb_expanding = False
        if not pd.isna(bb_w) and len(df) > 5:
            prev_bb = df["BB_Width"].iloc[-5]
            bb_expanding = not pd.isna(prev_bb) and bb_w > prev_bb * 1.2
        r.checks.append(CheckItem("BB 스퀴즈 해소", bb_expanding,
                                   f"{bb_w:.3f}" if not pd.isna(bb_w) else "-", 1.0))

        # 진입 시그널
        if ma_aligned and sma200_rising:
            if near_high and vol_ok:
                r.entry_signals.append("피봇 돌파 (고점 근접 + 거래량)")
            if bb_expanding and vol_ok:
                r.entry_signals.append("변동성 확장 돌파")
            if rs_ok and near_high:
                r.entry_signals.append("RS 강세 + 신고가 영역")

        # 손절가 (7.5% 고정 — Mark Minervini 규칙)
        r.stop_loss = round(last["Close"] * 0.925)
        # 목표가 (1차 +20%)
        r.target_price = round(last["Close"] * 1.20)

        return r
