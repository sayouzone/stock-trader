"""시장 레짐 감지 & 전략 추천 페이지"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _sma(s, p): return s.rolling(p, min_periods=p).mean()
def _ema(s, p): return s.ewm(span=p, adjust=False).mean()
def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/p, min_periods=p).mean()
    l = (-d).clip(lower=0).ewm(alpha=1/p, min_periods=p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _atr(df, p=14):
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-df["Close"].shift(1)).abs(),
                    (df["Low"]-df["Close"].shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, min_periods=p).mean()

def _adx(df, p=14):
    pdm = df["High"].diff().clip(lower=0)
    mdm = (-df["Low"].diff()).clip(lower=0)
    pdm = pdm.where(pdm > mdm, 0.0)
    mdm = mdm.where(mdm > pdm, 0.0)
    a = _atr(df, p)
    pdi = 100 * pdm.ewm(alpha=1/p, min_periods=p).mean() / a
    mdi = 100 * mdm.ewm(alpha=1/p, min_periods=p).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/p, min_periods=p).mean()


def _generate_scenario(scenario, days=750):
    rng = np.random.default_rng(42)
    seg = days // 4
    configs = {
        "강세장": (np.full(days, 0.0006), np.full(days, 0.015)),
        "약세장": (np.full(days, -0.0008), np.full(days, 0.025)),
        "고변동성": (np.full(days, 0.0001), np.full(days, 0.035)),
        "혼합장": (
            np.concatenate([np.full(seg, 0.0008), np.full(seg, 0.0001),
                            np.full(seg, -0.001), np.full(days-3*seg, 0.0006)]),
            np.concatenate([np.full(seg, 0.012), np.full(seg, 0.018),
                            np.full(seg, 0.028), np.full(days-3*seg, 0.020)])
        ),
    }
    mu_seq, sigma_seq = configs[scenario]
    prices = [100.0]
    for i in range(days - 1):
        prices.append(prices[-1] * np.exp(mu_seq[i] + sigma_seq[i] * rng.standard_normal()))
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)
    c = np.array(prices)
    return pd.DataFrame({
        "Open": c * (1 + rng.uniform(-0.01, 0.01, days)),
        "High": c * (1 + rng.uniform(0.002, 0.025, days)),
        "Low": c * (1 - rng.uniform(0.002, 0.025, days)),
        "Close": c, "Volume": rng.integers(1_000_000, 10_000_000, days).astype(float),
    }, index=dates)


def _detect_regime(df):
    c = df["Close"]
    sma200 = _sma(c, 200)
    sma50 = _sma(c, 50)
    sma150 = _sma(c, 150)
    rsi_v = _rsi(c)
    adx_v = _adx(df)
    macd_l = _ema(c, 12) - _ema(c, 26)
    macd_s = _ema(macd_l, 9)
    macd_h = macd_l - macd_s

    scores = []
    for i in range(200, len(df)):
        s = 0
        w_total = 0
        # 200MA
        if not pd.isna(sma200.iloc[i]) and sma200.iloc[i] > 0:
            pct = (c.iloc[i] - sma200.iloc[i]) / sma200.iloc[i]
            sc = 2 if pct > 0.1 else 1 if pct > 0.02 else 0 if pct > -0.02 else -1 if pct > -0.1 else -2
            s += sc * 2; w_total += 2
        # MA배열
        if all(not pd.isna(v) for v in [sma50.iloc[i], sma150.iloc[i], sma200.iloc[i]]):
            if sma50.iloc[i] > sma150.iloc[i] > sma200.iloc[i]: sc = 2
            elif sma50.iloc[i] > sma200.iloc[i]: sc = 1
            elif sma50.iloc[i] < sma150.iloc[i] < sma200.iloc[i]: sc = -2
            elif sma50.iloc[i] < sma200.iloc[i]: sc = -1
            else: sc = 0
            s += sc * 1.5; w_total += 1.5
        # RSI
        r = rsi_v.iloc[i]
        if not pd.isna(r):
            sc = 1.5 if r > 70 else 1 if r > 55 else 0 if r > 45 else -1 if r > 30 else -1.5
            s += sc; w_total += 1
        # MACD
        if not pd.isna(macd_h.iloc[i]) and not pd.isna(macd_l.iloc[i]):
            if macd_l.iloc[i] > 0 and macd_h.iloc[i] > 0: sc = 1.5
            elif macd_l.iloc[i] > 0: sc = 0.5
            elif macd_l.iloc[i] < 0 and macd_h.iloc[i] < 0: sc = -1.5
            elif macd_l.iloc[i] < 0: sc = -0.5
            else: sc = 0
            s += sc; w_total += 1

        scores.append(s / w_total if w_total > 0 else 0)

    return pd.Series(scores, index=df.index[200:]), sma200, sma50


STRATEGY_RECS = {
    "강한 강세": [("성장주", 5, "피봇 돌파 + 거래량 급증"), ("모멘텀", 5, "듀얼 모멘텀 최적"),
                 ("포지션", 4, "200MA 위 + Stage 2"), ("스윙", 3, "풀백 매수"), ("가치", 2, "기회비용 큼"), ("배당", 2, "상대수익률 열위")],
    "강세":     [("포지션", 5, "MA 풀백 최적"), ("모멘텀", 4, "추세 추종"), ("성장주", 4, "MA 정배열"),
                 ("스윙", 3, "롱 방향만"), ("가치", 2, "저평가 희소"), ("배당", 2, "DCA 축소")],
    "횡보":     [("스윙", 5, "RSI/BB 범위 매매"), ("배당", 4, "인컴 수익"), ("가치", 4, "골든크로스 대기"),
                 ("포지션", 2, "시그널 노이즈"), ("성장주", 2, "휩소 빈번"), ("모멘텀", 1, "추세 부재")],
    "약세":     [("배당", 5, "하방 보호 + DCA"), ("가치", 4, "저점 매수 기회"), ("스윙", 3, "숏 방향"),
                 ("포지션", 1, "매매 금지"), ("성장주", 1, "매매 중단"), ("모멘텀", 1, "현금 전환")],
    "강한 약세": [("배당", 4, "고품질만 DCA"), ("가치", 3, "분할 매수"), ("스윙", 2, "극히 제한"),
                 ("포지션", 1, "현금 대기"), ("성장주", 1, "전면 중단"), ("모멘텀", 1, "100% 현금")],
}


def render():
    st.markdown("## 🌡️ 시장 레짐 감지기")
    st.markdown("6가지 지표로 현재 시장 국면을 판별하고 최적 전략을 추천합니다.")

    col_s1, col_s2 = st.columns(2)
    scenario = col_s1.selectbox("시나리오", ["혼합장", "강세장", "약세장", "고변동성"])
    days = col_s2.slider("분석 기간 (일)", 400, 1000, 750, 50)

    if st.button("🔍 레짐 분석 실행", type="primary"):
        with st.spinner("분석 중..."):
            df = _generate_scenario(scenario, days)
            scores, sma200, sma50 = _detect_regime(df)

        # 현재 레짐 판정
        current_score = scores.iloc[-1]
        if current_score >= 1.2: regime = "강한 강세"
        elif current_score >= 0.4: regime = "강세"
        elif current_score >= -0.4: regime = "횡보"
        elif current_score >= -1.2: regime = "약세"
        else: regime = "강한 약세"

        regime_colors = {"강한 강세": "#00C853", "강세": "#69F0AE", "횡보": "#FFD54F",
                         "약세": "#FF8A80", "강한 약세": "#D32F2F"}
        regime_icons = {"강한 강세": "🟢", "강세": "🔵", "횡보": "🟡", "약세": "🔴", "강한 약세": "⚫"}

        # 상단 상태
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card" style="background:{regime_colors[regime]}"><h3>현재 레짐</h3><p>{regime_icons[regime]} {regime}</p></div>', unsafe_allow_html=True)
        c2.metric("종합 점수", f"{current_score:+.2f}", "(-2.0 약세 ~ +2.0 강세)")
        c3.metric("확신도", f"{min(abs(current_score)/2*100, 100):.0f}%")

        st.divider()

        # 가격 + 레짐 차트
        st.markdown("### 📈 레짐 타임라인")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                            vertical_spacing=0.05)

        # 가격
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="종가",
                                 line=dict(color="black", width=1.5)), row=1, col=1)
        if sma200 is not None:
            fig.add_trace(go.Scatter(x=df.index, y=sma200, name="200MA",
                                     line=dict(color="blue", width=1, dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=sma50, name="50MA",
                                     line=dict(color="orange", width=1, dash="dot")), row=1, col=1)

        # 레짐 배경 (간소화)
        colors_bar = ["#00C853" if s >= 1.2 else "#69F0AE" if s >= 0.4 else "#FFD54F" if s >= -0.4
                      else "#FF8A80" if s >= -1.2 else "#D32F2F" for s in scores]
        fig.add_trace(go.Bar(x=scores.index, y=scores, marker_color=colors_bar,
                             name="레짐 점수", opacity=0.8), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        fig.update_layout(height=600, legend=dict(x=0.01, y=0.99))
        fig.update_yaxes(title_text="가격", row=1, col=1)
        fig.update_yaxes(title_text="레짐 점수", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # 레짐 분포
        st.markdown("### 📊 레짐 분포")
        regime_labels = []
        for s in scores:
            if s >= 1.2: regime_labels.append("강한 강세")
            elif s >= 0.4: regime_labels.append("강세")
            elif s >= -0.4: regime_labels.append("횡보")
            elif s >= -1.2: regime_labels.append("약세")
            else: regime_labels.append("강한 약세")

        regime_counts = pd.Series(regime_labels).value_counts()
        fig2 = go.Figure(data=go.Pie(
            labels=regime_counts.index, values=regime_counts.values,
            marker_colors=[regime_colors[r] for r in regime_counts.index],
            hole=0.4, textinfo="label+percent"
        ))
        fig2.update_layout(height=350)

        col_a, col_b = st.columns([1, 1.5])
        col_a.plotly_chart(fig2, use_container_width=True)

        # 전략 추천
        with col_b:
            st.markdown(f"### 🎯 전략 추천 ({regime_icons[regime]} {regime})")
            recs = STRATEGY_RECS.get(regime, [])
            for strat, score, reason in recs:
                stars = "★" * score + "☆" * (5 - score)
                color = "green" if score >= 4 else "orange" if score >= 3 else "red"
                st.markdown(f":{color}[**{stars}** {strat}] — {reason}")

        st.divider()

        # 전환 이벤트
        st.markdown("### 🔄 레짐 전환 이벤트")
        transitions = []
        for i in range(1, len(regime_labels)):
            if regime_labels[i] != regime_labels[i-1]:
                transitions.append({
                    "날짜": scores.index[i].strftime("%Y-%m-%d"),
                    "전환": f"{regime_labels[i-1]} → {regime_labels[i]}",
                    "점수": f"{scores.iloc[i]:+.2f}",
                })
        if transitions:
            st.dataframe(pd.DataFrame(transitions[-15:]), use_container_width=True, hide_index=True)
            st.caption(f"총 {len(transitions)}회 전환 중 최근 15건 표시")
        else:
            st.info("전환 이벤트 없음")
    else:
        st.info("👆 '레짐 분석 실행' 버튼을 클릭하면 분석이 시작됩니다.")
