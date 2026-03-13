"""홈 페이지 — 종합 요약 대시보드"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render():
    st.markdown("## 🏠 주식투자 종합 대시보드")
    st.markdown("**'주식투자기법 종합가이드'** 문서 기반 — 6가지 전략의 포지션 사이징, 리스크 관리, 포트폴리오 최적화를 한 곳에서.")

    # 상단 메트릭 카드
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card blue-card"><h3>총 자산</h3><p>₩100,000,000</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card green-card"><h3>투자금</h3><p>₩52,381,420</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card yellow-card"><h3>현금 비중</h3><p>47.6%</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card red-card"><h3>총 리스크</h3><p>3.05%</p></div>', unsafe_allow_html=True)

    st.divider()

    # 6전략 요약
    st.markdown("### 📋 6가지 투자 전략 요약")
    strategies = pd.DataFrame({
        "전략": ["포지션", "성장주", "가치", "스윙", "모멘텀", "배당"],
        "목표 승률": ["40~50%", "35~45%", "55~65%", "50~60%", "45~55%", "65~75%"],
        "목표 손익비": ["3:1+", "3:1~5:1", "2:1~3:1", "1.5:1~2:1", "2:1", "1.5:1"],
        "보유 기간": ["수주~수개월", "수주~수개월", "수개월~수년", "2~8일", "1개월+", "장기"],
        "핵심 지표": ["200MA, ADX, Stage 2", "CAN SLIM, 피봇", "PER/PBR, 골든크로스", "RSI, BB, 스토캐스틱", "듀얼 모멘텀", "DCA, 수익률 밴드"],
        "시장 환경": ["강세", "강한 강세", "횡보/약세 후반", "횡보", "강세", "약세/횡보"],
    })
    st.dataframe(strategies, use_container_width=True, hide_index=True)

    st.divider()

    # 리스크 관리 규칙 요약
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🛡️ 리스크 관리 규칙")
        st.markdown("""
        | 규칙 | 조건 | 행동 |
        |------|------|------|
        | 연패 규칙 | 3연패 | 포지션 사이즈 50% 축소 |
        | 매매 중단 | 5연패 | 즉시 매매 중단, 전략 점검 |
        | 월간 한도 | 월 -6% | 해당 월 매매 중단 |
        | 포지션 사이징 | 항상 | (계좌×리스크%) ÷ (진입-손절) |
        """)

    with col2:
        st.markdown("### 📊 포지션 사이징 공식")
        st.latex(r"\text{매수 수량} = \frac{\text{계좌 자산} \times \text{리스크\%}}{\text{진입가} - \text{손절가}}")
        st.info("매매당 리스크는 계좌의 1~2%를 권장합니다. 연패 시 자동으로 50% 축소됩니다.")

    st.divider()

    # 시장 레짐 + 전략 적합도 히트맵
    st.markdown("### 🌡️ 시장 레짐별 전략 적합도")
    regimes = ["강한 강세", "강세", "횡보", "약세", "강한 약세"]
    strats = ["포지션", "성장주", "가치", "스윙", "모멘텀", "배당"]
    matrix = np.array([
        [4, 5, 2, 3, 5, 2],
        [5, 4, 2, 3, 4, 2],
        [2, 2, 4, 5, 1, 4],
        [1, 1, 4, 3, 1, 5],
        [1, 1, 3, 2, 1, 4],
    ])

    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=strats, y=regimes,
        colorscale=[[0, "#FF6B6B"], [0.5, "#FFE066"], [1, "#69DB7C"]],
        text=matrix, texttemplate="%{text}★", textfont={"size": 14},
        zmin=1, zmax=5,
        colorbar=dict(title="적합도", tickvals=[1, 2, 3, 4, 5],
                      ticktext=["회피", "부적합", "보통", "우수", "최적"]),
    ))
    fig.update_layout(height=350, margin=dict(t=30, b=30), yaxis_autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
