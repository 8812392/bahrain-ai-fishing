import streamlit as st
import pandas as pd

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Sustainable Fishing in Bahrain",
    page_icon="🌊",
    layout="wide"
)

# -------------------------------------------------
# GLOBAL STYLES (LIGHT OCEAN THEME)
# -------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #f0f9ff;
    color: #0b2b3c;
}

h1, h2, h3 {
    color: #083344;
}

.main-card {
    background: white;
    border-radius: 18px;
    padding: 32px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.06);
    margin-bottom: 30px;
}

.badge {
    display: inline-block;
    background: #e0f2fe;
    color: #0369a1;
    padding: 8px 16px;
    border-radius: 999px;
    margin-right: 10px;
    font-weight: 600;
    font-size: 14px;
}

.result-good {
    background: #dcfce7;
    color: #065f46;
    padding: 16px;
    border-radius: 12px;
    font-weight: 600;
}

.result-bad {
    background: #fee2e2;
    color: #7f1d1d;
    padding: 16px;
    border-radius: 12px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("""
<div class="main-card">
<h1>AI Tool for Sustainable Fishing in Bahrain 🇧🇭</h1>
<p style="font-size:18px;">
An educational Streamlit application that explores how fishing practices
can impact marine sustainability in Bahrain and the Gulf region.
</p>

<span class="badge">🌊 Ocean-themed UI</span>
<span class="badge">🤖 AI-inspired logic</span>
<span class="badge">🎓 Student project</span>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# DATA LOADING (FAIL-SAFE)
# -------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("bahrain_fishing_practices.csv")
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception:
        return None

df = load_data()

# -------------------------------------------------
# MAIN TOOL
# -------------------------------------------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.header("Fishing Practice Sustainability Checker")

if df is None:
    st.warning("Dataset not available. Demo mode enabled.")

    fishing_method = st.selectbox(
        "Fishing method",
        ["Hand line", "Trap", "Net", "Trawl"]
    )

    gear_type = st.selectbox(
        "Gear type",
        ["Selective", "Non-selective"]
    )
else:
    fishing_method = st.selectbox(
        "Fishing method",
        sorted(df.get("fishing_method", pd.Series(["Hand line"])).unique())
    )

    gear_type = st.selectbox(
        "Gear type",
        sorted(df.get("gear_type", pd.Series(["Selective"])).unique())
    )

catch_amount = st.slider(
    "Estimated catch per trip (kg)",
    min_value=1,
    max_value=1000,
    value=150
)

if st.button("Predict sustainability"):
    score = 0

    if "hand" in fishing_method.lower():
        score += 1
    if "selective" in gear_type.lower():
        score += 1
    if catch_amount <= 300:
        score += 1

    if score >= 2:
        st.markdown(
            "<div class='result-good'>✅ This practice is likely sustainable.</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-bad'>⚠️ This practice may be unsustainable.</div>",
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# FOOTER NOTE
# -------------------------------------------------
st.info("Use the sidebar to read *My Story* and learn why this project matters personally.")

