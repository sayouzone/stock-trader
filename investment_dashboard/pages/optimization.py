"""포트폴리오 최적화 & 효율적 프론티어 페이지"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _generate_returns(n_assets=8, n_days=500, seed=42):
    rng = np.random.default_rng(seed)
    names = ["삼성전자", "SK하이닉스", "NAVER", "카카오",
             "LG에너지", "셀트리온", "현대차", "KB금융"][:n_assets]
    corr = np.eye(n_assets)
    for i in range(min(4, n_assets)):
        for j in range(i + 1, min(4, n_assets)):
            corr[i, j] = corr[j, i] = rng.uniform(0.4, 0.7)
    for i in range(4, n_assets):
        for j in range(i + 1, n_assets):
            corr[i, j] = corr[j, i] = rng.uniform(-0.1, 0.3)
    L = np.linalg.cholesky(corr + np.eye(n_assets) * 0.01)
    raw = rng.standard_normal((n_days, n_assets))
    correlated = raw @ L.T
    mus = rng.uniform(0.0001, 0.0008, n_assets)
    sigmas = rng.uniform(0.015, 0.035, n_assets)
    returns = mus + correlated * sigmas
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    return pd.DataFrame(returns, index=dates, columns=names)


def _optimize(returns, rf=0.035, n_sim=15000):
    n = returns.shape[1]
    names = list(returns.columns)
    mean_r = returns.mean() * 252
    cov = returns.cov() * 252
    rng = np.random.default_rng(42)

    results = []
    for _ in range(n_sim):
        w = rng.random(n)
        w /= w.sum()
        ret = w @ mean_r.values
        risk = np.sqrt(w @ cov.values @ w)
        sharpe = (ret - rf) / risk if risk > 0 else 0
        results.append({"return": ret, "risk": risk, "sharpe": sharpe,
                        **{f"w_{name}": w[i] for i, name in enumerate(names)}})
    df = pd.DataFrame(results)

    # 최적 포트폴리오들
    idx_max_sharpe = df["sharpe"].idxmax()
    idx_min_var = df["risk"].idxmin()

    # 리스크 패리티
    w_rp = np.ones(n) / n
    for _ in range(300):
        pv = w_rp @ cov.values @ w_rp
        pvol = np.sqrt(pv)
        mrc = cov.values @ w_rp / pvol
        rc = w_rp * mrc
        trc = pvol / n
        w_new = w_rp * trc / (rc + 1e-12)
        w_new = np.maximum(w_new, 0)
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w_rp)) < 1e-8:
            break
        w_rp = w_new
    rp_ret = w_rp @ mean_r.values
    rp_risk = np.sqrt(w_rp @ cov.values @ w_rp)

    # 켈리
    w_kelly = np.zeros(n)
    for i, name in enumerate(names):
        rets = returns[name].dropna()
        wins, losses = rets[rets > 0], rets[rets < 0]
        if len(wins) > 0 and len(losses) > 0:
            p = len(wins) / len(rets)
            b = wins.mean() / abs(losses.mean())
            w_kelly[i] = max(0, (b * p - (1 - p)) / b * 0.5)
    if w_kelly.sum() > 0:
        w_kelly /= w_kelly.sum()
    else:
        w_kelly = np.ones(n) / n
    k_ret = w_kelly @ mean_r.values
    k_risk = np.sqrt(w_kelly @ cov.values @ w_kelly)

    return df, names, mean_r, cov, {
        "max_sharpe": {"w": {names[i]: df.iloc[idx_max_sharpe][f"w_{names[i]}"] for i in range(n)},
                       "ret": df.iloc[idx_max_sharpe]["return"], "risk": df.iloc[idx_max_sharpe]["risk"],
                       "sharpe": df.iloc[idx_max_sharpe]["sharpe"]},
        "min_var": {"w": {names[i]: df.iloc[idx_min_var][f"w_{names[i]}"] for i in range(n)},
                    "ret": df.iloc[idx_min_var]["return"], "risk": df.iloc[idx_min_var]["risk"],
                    "sharpe": df.iloc[idx_min_var]["sharpe"]},
        "risk_parity": {"w": dict(zip(names, w_rp)), "ret": rp_ret, "risk": rp_risk,
                        "sharpe": (rp_ret - rf) / rp_risk if rp_risk > 0 else 0},
        "kelly": {"w": dict(zip(names, w_kelly)), "ret": k_ret, "risk": k_risk,
                  "sharpe": (k_ret - rf) / k_risk if k_risk > 0 else 0},
    }


def render():
    st.markdown("## 📊 포트폴리오 최적화")
    st.markdown("4가지 최적화 방법으로 최적 비중을 산출하고 효율적 프론티어를 시각화합니다.")

    col_s1, col_s2 = st.columns(2)
    n_assets = col_s1.slider("종목 수", 4, 8, 8)
    n_sim = col_s2.slider("시뮬레이션 수", 5000, 30000, 15000, 5000)

    if st.button("🚀 최적화 실행", type="primary"):
        with st.spinner("최적화 중..."):
            returns = _generate_returns(n_assets)
            sim_df, names, mean_r, cov, opts = _optimize(returns, n_sim=n_sim)

        # 효율적 프론티어
        st.markdown("### 📈 효율적 프론티어")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sim_df["risk"] * 100, y=sim_df["return"] * 100,
            mode="markers", marker=dict(size=3, color=sim_df["sharpe"], colorscale="RdYlGn",
                                        colorbar=dict(title="샤프")),
            name="랜덤 포트폴리오", opacity=0.5
        ))

        labels = {"max_sharpe": "최대 샤프", "min_var": "최소 분산",
                  "risk_parity": "리스크 패리티", "kelly": "켈리 기준"}
        colors = {"max_sharpe": "#2196F3", "min_var": "#F44336",
                  "risk_parity": "#4CAF50", "kelly": "#FF9800"}
        symbols = {"max_sharpe": "star", "min_var": "diamond",
                   "risk_parity": "triangle-up", "kelly": "circle"}

        for key, opt in opts.items():
            fig.add_trace(go.Scatter(
                x=[opt["risk"] * 100], y=[opt["ret"] * 100],
                mode="markers+text",
                marker=dict(size=18, color=colors[key], symbol=symbols[key],
                            line=dict(width=2, color="black")),
                text=[f"{labels[key]}<br>S={opt['sharpe']:.2f}"],
                textposition="top center", name=labels[key],
            ))

        fig.update_layout(
            xaxis_title="리스크 (연율 %)", yaxis_title="수익률 (연율 %)",
            height=500, legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig, use_container_width=True)

        # 비교 테이블
        st.markdown("### 📋 최적화 방법 비교")
        comp_data = []
        for key, opt in opts.items():
            comp_data.append({
                "방법": labels[key],
                "기대 수익률": f"{opt['ret']:.1%}",
                "리스크": f"{opt['risk']:.1%}",
                "샤프 비율": f"{opt['sharpe']:.2f}",
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        # 비중 비교
        st.markdown("### 🎯 비중 비교")
        fig2 = make_subplots(rows=1, cols=4, specs=[[{"type": "pie"}] * 4],
                             subplot_titles=[labels[k] for k in opts.keys()])
        for i, (key, opt) in enumerate(opts.items()):
            w = opt["w"]
            filtered = {k: v for k, v in w.items() if v > 0.01}
            fig2.add_trace(go.Pie(
                labels=list(filtered.keys()), values=list(filtered.values()),
                textinfo="label+percent", textposition="inside",
                hole=0.3
            ), row=1, col=i + 1)
        fig2.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        # 상관관계
        st.markdown("### 🔗 상관관계 히트맵")
        corr = returns.corr()
        fig3 = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[[0, "#4444FF"], [0.5, "#FFFFFF"], [1, "#FF4444"]],
            zmin=-1, zmax=1, text=np.round(corr.values, 2), texttemplate="%{text}",
        ))
        fig3.update_layout(height=450)
        st.plotly_chart(fig3, use_container_width=True)

        # 리스크 기여도
        st.markdown("### ⚖️ 리스크 기여도 (최대 샤프)")
        best = opts["max_sharpe"]
        w_arr = np.array([best["w"][n] for n in names])
        pv = w_arr @ cov.values @ w_arr
        pvol = np.sqrt(pv)
        mrc = cov.values @ w_arr / pvol
        rc = w_arr * mrc
        rc_pct = rc / rc.sum() * 100 if rc.sum() > 0 else rc

        fig4 = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]],
                             subplot_titles=["비중 배분", "리스크 기여도"])
        filtered_w = {n: w for n, w in best["w"].items() if w > 0.01}
        fig4.add_trace(go.Pie(labels=list(filtered_w.keys()), values=list(filtered_w.values()),
                              hole=0.3), row=1, col=1)
        fig4.add_trace(go.Bar(x=names, y=rc_pct, marker_color="#2196F3"), row=1, col=2)
        fig4.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("👆 '최적화 실행' 버튼을 클릭하면 몬테카를로 시뮬레이션이 시작됩니다.")
