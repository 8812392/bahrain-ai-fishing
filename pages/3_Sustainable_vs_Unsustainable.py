import streamlit as st

st.set_page_config(page_title="Fishing Practices Guide", page_icon="🌊", layout="wide")

OCEAN_CSS = """
<style>
.stApp {
  background: linear-gradient(180deg, #F7FCFF 0%, #FFFFFF 40%, #F2FBFF 100%);
  color: #0B2233;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.block-container { padding-top: 1.2rem; }
.container { max-width: 1080px; margin: 0 auto; }

.hero {
  background: radial-gradient(1200px 420px at 20% 0%, rgba(24, 163, 200, 0.18), rgba(255,255,255,0) 55%),
              linear-gradient(180deg, #FFFFFF 0%, #F7FCFF 100%);
  border: 1px solid rgba(2, 132, 199, 0.14);
  border-radius: 20px;
  padding: 22px;
  box-shadow: 0 12px 28px rgba(11, 34, 51, 0.06);
}
.hero h1 { margin: 0; font-size: 42px; letter-spacing: -0.02em; }
.hero p  { margin: 10px 0 0 0; font-size: 16.5px; line-height: 1.6; color: rgba(11,34,51,0.78); }

.card {
  background: rgba(255,255,255,0.92);
  border: 1px solid rgba(2, 132, 199, 0.14);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 10px 24px rgba(11, 34, 51, 0.05);
}
.card h2 { margin: 0 0 10px 0; font-size: 26px; letter-spacing: -0.01em; }
.card p, .card li { color: rgba(11,34,51,0.82); font-size: 15.5px; line-height: 1.6; }
.muted { color: rgba(11,34,51,0.62); font-size: 13.5px; }

.pill {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 999px;
  border: 1px solid rgba(2,132,199,0.20);
  background: rgba(56, 189, 248, 0.10);
  font-weight: 700;
}
.pill.good { border-color: rgba(16,185,129,0.26); background: rgba(16,185,129,0.10); color: #064E3B; }
.pill.bad  { border-color: rgba(239,68,68,0.22); background: rgba(239,68,68,0.08); color: #7F1D1D; }
</style>
"""
st.markdown(OCEAN_CSS, unsafe_allow_html=True)

st.markdown('<div class="container">', unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>Fishing Practices Guide 🌊</h1>
  <p>
    This page explains the difference between <b>sustainable</b> and <b>unsustainable</b> fishing in a beginner-friendly way,
    with examples that fit Bahrain and the Gulf region.
  </p>
</div>
""", unsafe_allow_html=True)

st.write("")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
<div class="card">
  <div class="pill good">✅ Sustainable practices (better choices)</div>
  <ul style="margin-top: 12px;">
    <li><b>Selective gear</b> (like hook & line): catches target fish with less bycatch.</li>
    <li><b>Respecting size/season rules:</b> helps fish reproduce before being caught.</li>
    <li><b>Lower habitat impact:</b> avoids damaging coral, seagrass, and the seabed.</li>
    <li><b>Responsible catch sizes:</b> doesn’t remove too much from one area.</li>
    <li><b>Stronger enforcement:</b> rules are followed and monitored.</li>
  </ul>
  <p class="muted">These actions help keep stocks healthy for the future.</p>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown("""
<div class="card">
  <div class="pill bad">⚠️ Unsustainable practices (higher risk)</div>
  <ul style="margin-top: 12px;">
    <li><b>High bycatch gear</b> (some nets): can trap turtles, juveniles, and non-target species.</li>
    <li><b>Damaging habitats:</b> practices that drag or scrape the seabed can harm ecosystems.</li>
    <li><b>Overfishing pressure:</b> frequent high catches reduce stock levels fast.</li>
    <li><b>Weak enforcement:</b> rules exist but aren’t followed or monitored.</li>
    <li><b>Ghost gear</b> (lost nets/traps): keeps killing fish even after being lost.</li>
  </ul>
  <p class="muted">These patterns can reduce biodiversity and future catch.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

st.markdown("""
<div class="card">
  <h2>Bahrain examples (simple)</h2>
  <ul>
    <li><b>Better:</b> Smaller-scale line fishing with reasonable catch sizes, legal areas, and selective gear.</li>
    <li><b>Riskier:</b> Net-heavy methods in sensitive areas, very high catches, or weak monitoring.</li>
  </ul>
  <p class="muted">
    Note: This is educational guidance — always follow Bahrain’s official fishing regulations.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
