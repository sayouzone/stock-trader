"""전략 가이드 페이지 — 6가지 투자 전략 상세 설명"""
import streamlit as st
import plotly.graph_objects as go


STRATEGIES = {
    "📈 포지션 트레이딩": {
        "summary": "200MA 위 강세장에서 중장기 추세를 따라가는 전략",
        "win_rate": "40~50%", "rr": "3:1+", "period": "수주~수개월",
        "entry": [
            "① 시장 필터: 200MA 위 (강세장 확인)",
            "② ADX > 25 (추세 존재 확인)",
            "③ Stage 2 확인 (50MA 위 + 50MA 상승)",
            "④ 진입: MACD 시그널 교차 / 20MA 풀백 / 돌파",
        ],
        "exit": [
            "손절: 진입가 - ATR × 2.5 (후행 손절)",
            "1차 익절: +20% (포지션 1/3 청산)",
            "2차 익절: +50% (포지션 1/3 청산)",
            "최종: 50MA 이탈 시 전량 청산",
        ],
        "indicators": ["200MA", "50MA", "ADX", "ATR", "MACD", "Stage 분석"],
        "best_regime": "강세",
    },
    "🚀 성장주 트레이딩": {
        "summary": "CAN SLIM / SEPA 기반 피봇 돌파 매매",
        "win_rate": "35~45%", "rr": "3:1~5:1", "period": "수주~수개월",
        "entry": [
            "① MA 정배열: 50MA > 150MA > 200MA",
            "② 200MA 최소 1개월 상승 추세",
            "③ 52주 고점 근접 (90%+) — 피봇 영역",
            "④ 거래량 150%+ 돌파 시 매수",
        ],
        "exit": [
            "손절: 7~8% 고정 (Mark Minervini 규칙)",
            "1차 목표: +20% (포지션 1/2)",
            "2차 목표: +50% (잔여 포지션)",
            "손절가 라쳇: 수익 발생 시 손절가 올림",
        ],
        "indicators": ["50/150/200 MA", "거래량", "52주 고점", "BB Width", "RS 순위"],
        "best_regime": "강한 강세",
    },
    "💎 가치 트레이딩": {
        "summary": "저평가 종목의 재평가(리레이팅) 기회를 포착",
        "win_rate": "55~65%", "rr": "2:1~3:1", "period": "수개월~수년",
        "entry": [
            "① 52주 저점 근접 (저점 대비 +30% 이내)",
            "② 골든크로스 (50MA > 200MA) 확인",
            "③ MACD 0선 돌파 (추세 전환 시그널)",
            "④ PER/PBR 저평가 + Piotroski F-Score 8+",
        ],
        "exit": [
            "손절: 최근 저점 - ATR",
            "목표: 52주 고점 (리레이팅 완료)",
            "부분 익절: 골든크로스 후 +30% 시",
            "전량 청산: 데드크로스 발생 시",
        ],
        "indicators": ["PER/PBR", "골든크로스", "MACD", "F-Score", "ROE"],
        "best_regime": "횡보/약세 후반",
    },
    "⚡ 스윙 트레이딩": {
        "summary": "2~8일 단기 박스권 매매, 오실레이터 기반",
        "win_rate": "50~60%", "rr": "1.5:1~2:1", "period": "2~8일",
        "entry": [
            "① RSI 30 이하 과매도 → 반전 확인",
            "② 스토캐스틱 %K > %D 교차 (저점)",
            "③ BB 하단 터치 후 반등",
            "④ 20MA 풀백 반등",
        ],
        "exit": [
            "손절: ATR × 1.5",
            "목표: 2R (리스크의 2배)",
            "시간 정지: 8일 이내 미도달 시 청산",
            "BB 상단 도달 시 익절",
        ],
        "indicators": ["RSI", "스토캐스틱", "볼린저 밴드", "20MA", "ATR"],
        "best_regime": "횡보",
    },
    "🔥 모멘텀 트레이딩": {
        "summary": "듀얼 모멘텀 + 월별 리밸런싱 추세 추종",
        "win_rate": "45~55%", "rr": "2:1", "period": "1개월+",
        "entry": [
            "① 절대 모멘텀: 12개월 수익률 > 0",
            "② 상대 모멘텀: 유니버스 내 상위 랭킹",
            "③ 복합 스코어: 1M×0.3 + 3M×0.3 + 6M×0.25 + 12M×0.15",
            "④ 200MA 위 확인",
        ],
        "exit": [
            "절대 모멘텀 음수 → 100% 현금 전환",
            "월별 리밸런싱 시 하위 종목 교체",
            "손절: 10% 하락",
            "상대 모멘텀 하락 시 비중 축소",
        ],
        "indicators": ["1/3/6/12개월 수익률", "200MA", "ADX", "RS 순위"],
        "best_regime": "강세",
    },
    "💰 배당 투자": {
        "summary": "DCA + 수익률 밴드로 안정적 인컴 수익 추구",
        "win_rate": "65~75%", "rr": "1.5:1", "period": "장기",
        "entry": [
            "① 저변동성 확인 (ATR < 2%)",
            "② 수익률 밴드 하단 (가격 하위 35%)",
            "③ BB 하단 근접 시 추가 매수",
            "④ DCA: 정기 분할 매수",
        ],
        "exit": [
            "손절: 52주 저점 3% 하회 시",
            "수익률 밴드 상단 시 비중 축소",
            "배당컷 발생 시 전량 청산",
            "200MA 장기 하향 전환 시 축소",
        ],
        "indicators": ["배당수익률", "ATR", "BB", "200MA", "DCA 일정"],
        "best_regime": "약세/횡보",
    },
}


def render():
    st.markdown("## 📖 전략 가이드")
    st.markdown("**'주식투자기법 종합가이드'** 문서에 수록된 6가지 투자 전략의 상세 가이드입니다.")

    # 기대값 비교 차트
    st.markdown("### 🎲 전략별 기대값 비교")
    names = ["포지션", "성장주", "가치", "스윙", "모멘텀", "배당"]
    wrs = [0.45, 0.40, 0.60, 0.55, 0.50, 0.70]
    rrs = [3.0, 4.0, 2.5, 1.75, 2.0, 1.5]
    exps = [w * r - (1 - w) for w, r in zip(wrs, rrs)]

    fig = go.Figure()
    colors = ["#4CAF50" if e > 0.5 else "#2196F3" if e > 0 else "#F44336" for e in exps]
    fig.add_trace(go.Bar(x=names, y=exps, marker_color=colors,
                         text=[f"{e:+.2f}R" for e in exps], textposition="outside"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=300, yaxis_title="기대값 (R)", margin=dict(t=30))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 탭으로 각 전략 표시
    tabs = st.tabs(list(STRATEGIES.keys()))

    for tab, (name, info) in zip(tabs, STRATEGIES.items()):
        with tab:
            st.markdown(f"### {name}")
            st.markdown(f"**{info['summary']}**")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("목표 승률", info["win_rate"])
            c2.metric("목표 손익비", info["rr"])
            c3.metric("보유 기간", info["period"])
            c4.metric("최적 레짐", info["best_regime"])

            col_l, col_r = st.columns(2)

            with col_l:
                st.markdown("#### 🔵 진입 조건")
                for step in info["entry"]:
                    st.markdown(f"- {step}")

            with col_r:
                st.markdown("#### 🔴 청산 조건")
                for step in info["exit"]:
                    st.markdown(f"- {step}")

            st.markdown("**핵심 지표:** " + " · ".join(f"`{ind}`" for ind in info["indicators"]))

    st.divider()
    st.markdown("### 📐 공통 포지션 사이징 공식")
    st.latex(r"\text{매수 수량} = \frac{\text{계좌 자산} \times \text{리스크\%}}{\text{진입가} - \text{손절가}}")
    st.markdown("""
    | 리스크 관리 규칙 | 조건 | 행동 |
    |---|---|---|
    | 연패 규칙 | 3연패 | 포지션 사이즈 50% 축소 |
    | 매매 중단 | 5연패 | 즉시 매매 중단, 최소 2주 휴식 |
    | 월간 한도 | 월 -6% | 해당 월 잔여 기간 매매 중단 |
    | 최대 리스크 | 항상 | 매매당 1~2%, 포트폴리오 총 6% |
    """)
