import streamlit as st

st.set_page_config(
    page_title="How the Model Works",
    page_icon="🤖",
    layout="wide",
)

# Reuse the same ocean theme
OCEAN_CSS = """
<style>
.stApp {
  background: linear-gradient(180deg, #F7FCFF 0%, #FFFFFF 40%, #F2FBFF 100%);
  color: #0B2233;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.block-container { padding-top: 1.2rem; }

.container { max-width: 980px; margin: 0 auto; }

.header {
  background: radial-gradient(1000px 380px at 15% 0%, rgba(24, 163, 200, 0.18), rgba(255,255,255,0) 55%),
              linear-gradient(180deg, #FFFFFF 0%, #F7FCFF 100%);
  border: 1px solid rgba(2, 132, 199, 0.14);
  border-radius: 20px;
  padding: 22px;
  box-shadow: 0 12px 28px rgba(11, 34, 51, 0.06);
}

.card {
  background: rgba(255,255,255,0.92);
  border: 1px solid rgba(2, 132, 199, 0.14);
  border-radius: 18px;
  padding: 20px;
  box-shadow: 0 10px 24px rgba(11, 34, 51, 0.05);
}

.section-title {
  font-size: 26px;
  margin-bottom: 8px;
  letter-spacing: -0.01em;
}

.card p, .card li {
  font-size: 15.5px;
  line-height: 1.65;
  color: rgba(11,34,51,0.82);
}

.muted {
  font-size: 13.5px;
  color: rgba(11,34,51,0.62);
}
</style>
"""
st.markdown(OCEAN_CSS, unsafe_allow_html=True)

st.markdown('<div class="container">', unsafe_allow_html=True)

# Header
st.markdown(
    """
<div class="header">
  <h1>How the model works</h1>
  <p>
    This page explains what the model is doing — and just as importantly, what it is <b>not</b> doing.
    The goal is transparency, not complexity.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# Main content
st.markdown(
    """
<div class="card">
  <h2 class="section-title">What the model is trained on</h2>
  <p>
    The model is trained on example fishing practices from Bahrain and the Gulf region.
    Each example includes details such as fishing method, gear type, enforcement level,
    catch size, and species status — along with a sustainability label.
  </p>

  <h2 class="section-title" style="margin-top:18px;">How a prediction is made</h2>
  <p>
    When you enter a new fishing scenario, the model compares it to patterns it learned
    during training. It looks at how similar combinations of inputs were labeled in the past.
  </p>
  <p>
    The output is a probability-based estimate — not a rule, score, or judgment.
  </p>

  <h2 class="section-title" style="margin-top:18px;">Why the model is intentionally simple</h2>
  <p>
    A simpler model was chosen so that its behavior is easier to understand, explain,
    and improve. This makes it more suitable for education and discussion rather than
    black-box prediction.
  </p>

  <h2 class="section-title" style="margin-top:18px;">What the model does NOT do</h2>
  <ul>
    <li>It does not replace scientific stock assessments</li>
    <li>It does not enforce laws or regulations</li>
    <li>It does not account for seasonality, weather, or real-time data</li>
  </ul>

  <p class="muted">
    This model is a learning tool — its value comes from understanding patterns, not from treating outputs as facts.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)
