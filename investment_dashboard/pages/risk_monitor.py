"""리스크 모니터링 대시보드 페이지"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta


def _generate_trades(n=40, seed=123):
    rng = np.random.default_rng(seed)
    strategies = ["포지션", "성장주", "가치", "스윙", "모멘텀", "배당"]
    win_rates = [0.45, 0.40, 0.60, 0.55, 0.50, 0.70]
    rr_ratios = [3.0, 4.0, 2.5, 1.75, 2.0, 1.5]

    records = []
    base = date.today() - timedelta(days=n * 2)
    for i in range(n):
        idx = rng.integers(0, 6)
        strat = strategies[idx]
        wr, rr = win_rates[idx], rr_ratios[idx]
        is_win = rng.random() < wr
        r_mult = rng.uniform(0.5, rr * 1.5) if is_win else -rng.uniform(0.3, 1.2)
        pnl = r_mult * rng.uniform(500_000, 1_500_000)
        records.append({
            "날짜": base + timedelta(days=i * 2),
            "전략": strat,
            "종목": f"종목{i+1:02d}",
            "승패": "승" if is_win else "패",
            "R배수": round(r_mult, 2),
            "손익": round(pnl),
        })
    return pd.DataFrame(records)


def render():
    st.markdown("## 🛡️ 리스크 모니터링 대시보드")
    st.markdown("문서의 리스크 관리 규칙을 실시간으로 추적합니다.")

    trades = _generate_trades()

    # 상태 판정
    consec_losses = 0
    for _, r in trades.iloc[::-1].iterrows():
        if r["승패"] == "패":
            consec_losses += 1
        else:
            break

    month_trades = trades[trades["날짜"] >= date.today().replace(day=1)]
    month_pnl = month_trades["손익"].sum()
    month_return = month_pnl / 100_000_000

    total_pnl = trades["손익"].sum()
    wins = (trades["승패"] == "승").sum()
    total = len(trades)
    avg_r = trades["R배수"].mean()

    # 상단 상태 카드
    if consec_losses >= 5:
        status_html = '<div class="metric-card red-card"><h3>매매 상태</h3><p>🔴 중단</p></div>'
    elif consec_losses >= 3:
        status_html = '<div class="metric-card yellow-card"><h3>매매 상태</h3><p>⚠️ 50% 축소</p></div>'
    else:
        status_html = '<div class="metric-card green-card"><h3>매매 상태</h3><p>🟢 정상</p></div>'

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(status_html, unsafe_allow_html=True)
    c2.metric("현재 연패", f"{consec_losses}회", delta="3연패 시 50% 축소" if consec_losses >= 2 else None)
    c3.metric("이번 달 손익", f"₩{month_pnl:+,.0f}", f"{month_return:+.2%} / 한도 -6%")
    c4.metric("승률", f"{wins/total:.1%}", f"{wins}승 {total-wins}패")

    st.divider()

    # 규칙 체크리스트
    st.markdown("### ✅ 리스크 규칙 체크리스트")
    rules = [
        ("3연패 규칙", consec_losses < 3, f"현재 {consec_losses}연패" + (" → 50% 축소!" if consec_losses >= 3 else "")),
        ("5연패 규칙", consec_losses < 5, f"현재 {consec_losses}연패" + (" → 매매 중단!" if consec_losses >= 5 else "")),
        ("월간 손실 한도", month_return > -0.06, f"이번 달 {month_return:+.2%}" + (" → 매매 중단!" if month_return <= -0.06 else "")),
        ("매매당 리스크 ≤ 2%", True, "계좌의 1~2% 준수"),
        ("포트폴리오 리스크 ≤ 6%", True, "총 리스크 한도 이내"),
    ]
    for rule, ok, detail in rules:
        icon = "✅" if ok else "❌"
        color = "green" if ok else "red"
        st.markdown(f":{color}[{icon} **{rule}**] — {detail}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # 누적 손익 차트
        st.markdown("### 📈 누적 손익 곡선")
        trades_sorted = trades.sort_values("날짜")
        cum_pnl = trades_sorted["손익"].cumsum()

        fig = go.Figure()
        colors = ["green" if v >= 0 else "red" for v in cum_pnl]
        fig.add_trace(go.Scatter(
            x=trades_sorted["날짜"], y=cum_pnl,
            fill="tozeroy", fillcolor="rgba(76,175,80,0.1)",
            line=dict(color="#4CAF50", width=2), name="누적 손익"
        ))
        # 연패 구간 표시
        fig.update_layout(height=400, yaxis_title="누적 손익 (원)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # R배수 분포
        st.markdown("### 📊 R배수 분포")
        r_vals = trades["R배수"]
        colors = ["#4CAF50" if v > 0 else "#F44336" for v in r_vals]
        fig2 = go.Figure(data=go.Bar(
            x=list(range(len(r_vals))), y=r_vals,
            marker_color=colors
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.add_hline(y=r_vals.mean(), line_dash="dot", line_color="blue",
                       annotation_text=f"평균 {r_vals.mean():+.2f}R")
        fig2.update_layout(height=400, xaxis_title="매매 번호", yaxis_title="R배수")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # 전략별 분석
    st.markdown("### 🎯 전략별 성과")
    strat_stats = trades.groupby("전략").agg(
        매매수=("손익", "count"),
        승률=("승패", lambda x: (x == "승").mean()),
        총손익=("손익", "sum"),
        평균R=("R배수", "mean"),
    ).sort_values("총손익", ascending=False)
    strat_stats["승률"] = strat_stats["승률"].apply(lambda x: f"{x:.0%}")
    strat_stats["총손익"] = strat_stats["총손익"].apply(lambda x: f"₩{x:+,.0f}")
    strat_stats["평균R"] = strat_stats["평균R"].apply(lambda x: f"{x:+.2f}R")
    st.dataframe(strat_stats, use_container_width=True)

    # 전략별 승률 vs 손익비 차트
    strat_raw = trades.groupby("전략").agg(
        wr=("승패", lambda x: (x == "승").mean()),
        avg_win=("R배수", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
        avg_loss=("R배수", lambda x: abs(x[x < 0].mean()) if (x < 0).any() else 0),
    )
    strat_raw["rr"] = strat_raw["avg_win"] / strat_raw["avg_loss"].replace(0, 1)
    strat_raw["expectancy"] = strat_raw["wr"] * strat_raw["rr"] - (1 - strat_raw["wr"])

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=strat_raw["wr"] * 100, y=strat_raw["rr"],
        mode="markers+text", text=strat_raw.index,
        textposition="top center",
        marker=dict(size=strat_raw["expectancy"].clip(0) * 50 + 15,
                    color=strat_raw["expectancy"],
                    colorscale="RdYlGn", showscale=True,
                    colorbar=dict(title="기대값")),
    ))
    fig3.update_layout(
        title="전략별 승률 vs 손익비 (크기 = 기대값)",
        xaxis_title="승률 (%)", yaxis_title="손익비",
        height=400,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # 매매 기록 테이블
    st.markdown("### 📝 최근 매매 기록")
    st.dataframe(trades.sort_values("날짜", ascending=False).head(15),
                 use_container_width=True, hide_index=True)
