"""포지션 사이징 계산기 페이지"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render():
    st.markdown("## 📐 포지션 사이징 계산기")
    st.markdown("문서 공식: **매수 수량 = (계좌 × 리스크%) ÷ (진입가 - 손절가)**")

    st.divider()

    # 입력 패널
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("### 입력 값")
        capital = st.number_input("계좌 자산 (원)", value=100_000_000, step=10_000_000, format="%d")
        risk_pct = st.slider("매매당 리스크 (%)", 0.5, 3.0, 1.0, 0.1)
        entry_price = st.number_input("진입 예정가 (원)", value=72_000, step=1_000, format="%d")
        stop_loss = st.number_input("손절가 (원)", value=67_000, step=1_000, format="%d")

        strategy = st.selectbox("전략", [
            "포지션 (승률 40~50%, 손익비 3:1)",
            "성장주 (승률 35~45%, 손익비 3~5:1)",
            "가치 (승률 55~65%, 손익비 2~3:1)",
            "스윙 (승률 50~60%, 손익비 1.5~2:1)",
            "모멘텀 (승률 45~55%, 손익비 2:1)",
            "배당 (승률 65~75%, 손익비 1.5:1)",
        ])

        consecutive_losses = st.selectbox("현재 연패 상태", [
            "없음 (정상 사이즈)",
            "2연패 (주의)",
            "3연패 (50% 축소)",
            "4연패 (50% 축소)",
        ])

    # 계산
    risk_per_share = abs(entry_price - stop_loss)
    reduction = 0.5 if "3연패" in consecutive_losses or "4연패" in consecutive_losses else 1.0
    risk_amount = capital * (risk_pct / 100) * reduction

    if risk_per_share > 0:
        shares = int(risk_amount / risk_per_share)
        position_value = shares * entry_price
        position_pct = position_value / capital * 100
        actual_risk = shares * risk_per_share
        actual_risk_pct = actual_risk / capital * 100
    else:
        shares = position_value = position_pct = actual_risk = actual_risk_pct = 0

    with col2:
        st.markdown("### 계산 결과")

        r1, r2 = st.columns(2)
        r1.metric("매수 수량", f"{shares:,}주")
        r2.metric("투자금", f"₩{position_value:,.0f}")

        r3, r4 = st.columns(2)
        r3.metric("투자 비중", f"{position_pct:.1f}%")
        r4.metric("리스크 금액", f"₩{actual_risk:,.0f}")

        r5, r6 = st.columns(2)
        r5.metric("리스크 비율", f"{actual_risk_pct:.2f}%")
        r6.metric("리스크/주", f"₩{risk_per_share:,.0f}")

        if reduction < 1.0:
            st.warning(f"⚠️ 연패 규칙 적용: 포지션 사이즈 {reduction:.0%}로 축소됨")

        # 손익 시나리오
        strat_info = {
            "포지션": (3.0, 0.45), "성장주": (4.0, 0.40), "가치": (2.5, 0.60),
            "스윙": (1.75, 0.55), "모멘텀": (2.0, 0.50), "배당": (1.5, 0.70),
        }
        strat_key = strategy.split("(")[0].strip()
        rr, wr = strat_info.get(strat_key, (2.0, 0.50))

        target_price = entry_price + risk_per_share * rr
        profit_if_win = (target_price - entry_price) * shares
        loss_if_lose = risk_per_share * shares

        st.divider()
        st.markdown("### 손익 시나리오")
        s1, s2, s3 = st.columns(3)
        s1.metric("🎯 목표가", f"₩{target_price:,.0f}", f"+{(target_price/entry_price-1)*100:.1f}%")
        s2.metric("✅ 승리 시 이익", f"+₩{profit_if_win:,.0f}", f"+{rr:.1f}R")
        s3.metric("❌ 패배 시 손실", f"-₩{loss_if_lose:,.0f}", f"-1.0R")

    st.divider()

    # 시각화
    st.markdown("### 📊 리스크% 별 포지션 크기 분석")
    col_a, col_b = st.columns(2)

    with col_a:
        # 리스크 vs 포지션 크기
        risk_levels = [0.5, 1.0, 1.5, 2.0]
        stop_dists = np.arange(1, 11, 0.5)

        fig = go.Figure()
        colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
        for r, color in zip(risk_levels, colors):
            ra = capital * r / 100
            pos_pcts = [ra / (capital * d / 100) * 100 for d in stop_dists]
            fig.add_trace(go.Scatter(
                x=stop_dists, y=pos_pcts,
                name=f"리스크 {r}%", line=dict(width=2.5, color=color)
            ))

        # 현재 포지션 표시
        if risk_per_share > 0:
            stop_dist_pct = risk_per_share / entry_price * 100
            fig.add_trace(go.Scatter(
                x=[stop_dist_pct], y=[position_pct],
                mode="markers", marker=dict(size=15, color="red", symbol="star"),
                name="현재 설정"
            ))

        fig.update_layout(
            title="손절 거리 vs 포지션 크기",
            xaxis_title="손절 거리 (%)", yaxis_title="포지션 비중 (%)",
            height=400, yaxis_range=[0, 100]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # 연패별 리스크 금액
        scenarios = ["정상", "2연패", "3연패\n(50%)", "4연패\n(50%)", "5연패\n(중단)"]
        factors = [1.0, 1.0, 0.5, 0.5, 0.0]
        amounts = [capital * risk_pct / 100 * f for f in factors]
        colors = ["#4CAF50", "#FFC107", "#FF9800", "#FF5722", "#D32F2F"]

        fig = go.Figure(data=go.Bar(
            x=scenarios, y=amounts, marker_color=colors,
            text=[f"₩{a:,.0f}" for a in amounts], textposition="outside"
        ))
        fig.update_layout(
            title="연패 규칙에 따른 리스크 금액",
            yaxis_title="리스크 금액 (원)", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # 기대값 계산
    st.divider()
    st.markdown("### 🎲 기대값 분석")
    expectancy = wr * rr - (1 - wr)
    edge = expectancy * actual_risk if actual_risk > 0 else 0

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("승률", f"{wr:.0%}")
    e2.metric("손익비", f"{rr:.1f}:1")
    e3.metric("기대값 (R)", f"{expectancy:+.2f}R")
    e4.metric("매매당 기대수익", f"₩{edge:+,.0f}")

    if expectancy > 0:
        st.success(f"✅ 양의 기대값 (+{expectancy:.2f}R) — 100회 매매 시 예상 수익: ₩{edge * 100:+,.0f}")
    else:
        st.error(f"❌ 음의 기대값 ({expectancy:.2f}R) — 이 전략/설정은 장기적으로 손실 예상")
