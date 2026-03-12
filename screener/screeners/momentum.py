"""
모멘텀 트레이딩 스크리너
- 듀얼 모멘텀: 절대 + 상대 모멘텀
- 1/3/6/12개월 수익률 복합 스코어
- 월별 리밸런싱 기반
"""
import pandas as pd
from scoring import ScreenResult, CheckItem


class MomentumScreener:
    name = "모멘텀 트레이딩"

    @staticmethod
    def screen(ticker: str, df: pd.DataFrame) -> ScreenResult:
        r = ScreenResult(ticker=ticker, strategy="모멘텀 트레이딩", current_price=df["Close"].iloc[-1])
        last = df.iloc[-1]

        # 1. 절대 모멘텀 (12개월 수익률 > 0)
        ret_252 = last.get("Return_252d")
        abs_mom = not pd.isna(ret_252) and ret_252 > 0
        r.checks.append(CheckItem("12개월 수익률 > 0", abs_mom,
                                   f"{ret_252:.1%}" if not pd.isna(ret_252) else "-", 2.0))

        # 2. 6개월 수익률 양수
        ret_126 = last.get("Return_126d")
        ret6_ok = not pd.isna(ret_126) and ret_126 > 0
        r.checks.append(CheckItem("6개월 수익률 > 0", ret6_ok,
                                   f"{ret_126:.1%}" if not pd.isna(ret_126) else "-", 1.5))

        # 3. 복합 모멘텀 스코어 (가중 평균: 1M×0.3 + 3M×0.3 + 6M×0.25 + 12M×0.15)
        ret_21 = last.get("Return_21d", 0)
        ret_63 = last.get("Return_63d", 0)
        scores = []
        weights = []
        for ret, w in [(ret_21, 0.30), (ret_63, 0.30), (ret_126, 0.25), (ret_252, 0.15)]:
            if not pd.isna(ret):
                scores.append(ret * w)
                weights.append(w)
        composite = sum(scores) / sum(weights) if weights else 0
        composite_ok = composite > 0.05  # 5% 이상
        r.checks.append(CheckItem("복합 모멘텀 > 5%", composite_ok,
                                   f"{composite:.1%}", 2.0))

        # 4. 200MA 위 (상승 추세)
        sma200 = last.get("SMA_200")
        above200 = not pd.isna(sma200) and last["Close"] > sma200
        r.checks.append(CheckItem("200MA 위", above200,
                                   f"{'✓' if above200 else '✗'}", 1.5))

        # 5. ADX > 20 (추세 존재)
        adx_val = last.get("ADX", 0)
        adx_ok = not pd.isna(adx_val) and adx_val > 20
        r.checks.append(CheckItem("ADX > 20", adx_ok,
                                   f"{adx_val:.1f}" if not pd.isna(adx_val) else "-", 1.0))

        # 6. 하락 모멘텀 부재 (1개월 수익률 > -5%)
        ret_5 = last.get("Return_5d", 0)
        no_drop = not pd.isna(ret_21) and ret_21 > -0.05
        r.checks.append(CheckItem("1개월 낙폭 < 5%", no_drop,
                                   f"{ret_21:.1%}" if not pd.isna(ret_21) else "-", 1.0))

        # 진입 시그널
        if abs_mom and ret6_ok and composite_ok:
            r.entry_signals.append("듀얼 모멘텀 통과 (절대+복합)")
        if above200 and composite_ok and adx_ok:
            r.entry_signals.append("추세 모멘텀 확인")
        if not pd.isna(ret_21) and ret_21 > 0.10:
            r.entry_signals.append("단기 모멘텀 급등")

        # 손절가 (10% 하락)
        r.stop_loss = round(last["Close"] * 0.90)

        return r
