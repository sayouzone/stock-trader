"""
📊 주식투자 종합 대시보드
- 6가지 투자 전략 기반 포지션 사이징 / 리스크 관리 / 포트폴리오 최적화 / 시장 레짐 감지
- 문서 '주식투자기법 종합가이드' 기반
"""
import streamlit as st

st.set_page_config(
    page_title="주식투자 종합 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.2rem; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; opacity: 0.85; }
    .metric-card p { margin: 0; font-size: 1.6rem; font-weight: bold; }
    .green-card { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .red-card { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    .yellow-card { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); }
    .blue-card { background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px; border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── 사이드바 ──
st.sidebar.title("📊 투자 대시보드")
page = st.sidebar.radio("메뉴", [
    "🏠 홈",
    "📐 포지션 사이징",
    "📊 포트폴리오 최적화",
    "🛡️ 리스크 모니터링",
    "🌡️ 시장 레짐 감지",
    "📖 전략 가이드",
])

if page == "🏠 홈":
    import pages.home as m; m.render()
elif page == "📐 포지션 사이징":
    import pages.sizing as m; m.render()
elif page == "📊 포트폴리오 최적화":
    import pages.optimization as m; m.render()
elif page == "🛡️ 리스크 모니터링":
    import pages.risk_monitor as m; m.render()
elif page == "🌡️ 시장 레짐 감지":
    import pages.regime as m; m.render()
elif page == "📖 전략 가이드":
    import pages.strategy_guide as m; m.render()
