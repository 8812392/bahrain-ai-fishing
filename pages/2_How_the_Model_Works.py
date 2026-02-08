import streamlit as st

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="How the Model Works",
    page_icon="🤖",
    layout="wide",
)

# =========================
# Ocean theme CSS (same family as app + My Story)
# =========================
OCEAN_CSS = """
<style>
/* --- Global typography --- */
html, body, [class*="css"]  {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
}

/* App background */
.stApp {
  background: linear-gradient(180deg, #F7FCFF 0%, #FFFFFF 40%, #F2FBFF 100%);
  color: #0B2233;
}

/* Hide Streamlit menu/footer */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.block-container { padding-top: 1.2rem; }

/* Layout */
.container { max-width: 1080px; margin: 0 auto; }

/* Hero */
.hero {
  background: radial-gradient(1200px 420px at 20% 0%, rgba(24, 163, 200, 0.18), rgba(255,255,255,0) 55%),
              radial-gradient(900px 380px at 100% 20%, rgba(56, 189, 248, 0.14), rgba(255,255,255,0) 50%),
              linear-gradient(180deg, #FFFFFF 0%, #F7FCFF 100%);
  border: 1px solid rgba(2, 132, 199, 0.14);
  border-radius: 20px;
  padding: 22px 22px 18px 22px;
  box-shadow: 0 12px 28px rgba(11, 34, 51, 0.06);
}
.hero h1 {
  font-size: 42px;
  line-height: 1.1;
  margin: 0;
  letter-spacing: -0.02em;
}
.hero p {
  font-size: 16.5px;
  line-height: 1.6;
  margin: 10px 0 0 0;
  color: rgba(11, 34, 51, 0.78);
}

/* Badges */
.badges { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
.badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  padding: 7px 10px;
  border-radius: 999px;
  border: 1px solid rgba(2, 132, 199, 0.20);
  background: rgba(56, 189, 248, 0.10);
  color: rgba(11, 34, 51, 0.85);
}

/* Cards */
.card {
  background: rgba(255,255,255,0.92);
  border: 1px solid rgba(2, 132, 199, 0.14);
  border-radius: 18px;
  padding: 18px 18px 16px 18px;
  box-shadow: 0 10px 24px rgba(11, 34, 51, 0.05);
}
.card h2 { margin: 0 0 10px 0; font-size: 26px; letter-spacing: -0.01em; }
.card p, .card li { color: rgba(11, 34, 51, 0.82); font-size: 15.5px; line-height: 1.6; }

/* Section headers */
.section-title { font-size: 28px; margin: 0 0 10px 0; letter-spacing: -0.01em; }
.muted { color: rgba(11, 34, 51, 0.62); font-size: 13.5px; }

/* Diagram */
.flow {
  display: grid;
  grid-template-columns: 1fr auto 1fr auto 1fr;
  gap: 12px;
  align-items: stretch;
}
.flowbox {
  border-radius: 16px;
  border: 1px solid rgba(2, 132, 199, 0.18);
  background: rgba(56, 189, 248, 0.08);
  padding: 14px;
}
.flowbox h3 { margin: 0 0 6px 0; font-size: 16px; }
.flowbox p { margin: 0; font-size: 13.5px; color: rgba(11, 34, 51, 0.78); line-height: 1.5; }
.arrow {
  font-size: 22px;
  font-weight: 800;
  color: rgba(2, 132, 199, 0.75);
  align-self: center;
  padding: 0 2px;
}

/* Callouts */
.callout {
  border-radius: 16px;
  border: 1px solid rgba(2, 132, 199, 0.18);
  background: rgba(255, 255, 255, 0.92);
  padding: 14px;
}
.callout strong { color: #06324A; }

/* Inputs style */
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
  border-radius: 12px !important;
}
</style>
"""
st.markdown(OCEAN_CSS, unsafe_allow_html=True)

# =========================
# Content
# =========================
st.markdown('<div class="container">', unsafe_allow_html=True)

st.markdown(
    """
<div class="hero">
  <h1>How the model works</h1>
  <p>
    This page explains what the model is doing — and just as importantly, what it is <b>not</b> doing.
    The goal is transparency and trust, not complexity.
  </p>
  <div class="badges">
    <span class="badge">🧠 Simple + explainable</span>
    <span class="badge">🧼 Clean data pipeline</span>
    <span class="badge">🧭 Responsible use</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# --- Quick navigation (no clutter) ---
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**On this page**")
    with c2:
        st.markdown("• Flow diagram")
    with c3:
        st.markdown("• Training + prediction")
    with c4:
        st.markdown("• Limits + ethics")

st.write("")

# =========================
# Diagram: Input -> Pipeline -> Output
# =========================
st.markdown(
    """
<div class="card">
  <h2>Big picture (no jargon)</h2>
  <div class="flow" style="margin-top: 8px;">
    <div class="flowbox">
      <h3>1) You enter a scenario</h3>
      <p>Method, gear, species, enforcement, catch size, etc.</p>
    </div>
    <div class="arrow">➜</div>
    <div class="flowbox">
      <h3>2) The app prepares the inputs</h3>
      <p>Fills missing values + converts categories into numbers the model can understand.</p>
    </div>
    <div class="arrow">➜</div>
    <div class="flowbox">
      <h3>3) The model outputs an estimate</h3>
      <p>“Likely sustainable” or “likely unsustainable” — based on patterns in the example data.</p>
    </div>
  </div>

  <div class="callout" style="margin-top: 14px;">
    <strong>Important:</strong> this is a pattern-matching estimate from example data — not an official scientific judgment.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# =========================
# Main content columns
# =========================
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown(
        """
<div class="card">
  <h2>What the model is trained on</h2>
  <p>
    The model learns from a dataset of <b>example fishing practices</b>.
    Each row is one scenario (a combination of choices like method, gear, enforcement, catch size),
    and each scenario includes a <b>label</b> that says whether it is considered more likely sustainable or unsustainable.
  </p>
  <ul>
    <li><b>Inputs:</b> fishing method, target species, area type, gear type, enforcement level, catch per trip, and related factors.</li>
    <li><b>Output label:</b> sustainable vs unsustainable (the “answer” the model learns from).</li>
  </ul>
  <p class="muted">
    The model can only learn patterns that exist in the training examples. If the dataset is incomplete or biased,
    the predictions can also be incomplete or biased.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
<div class="card">
  <h2>How a prediction is made</h2>
  <p>
    When you submit a new scenario, the model compares it to patterns it learned during training.
    It does not “understand the ocean” — it recognizes combinations of inputs that, in the past,
    were associated with sustainable or unsustainable labels.
  </p>
  <ul>
    <li>It converts text categories (like gear type) into numeric signals.</li>
    <li>It uses a simple statistical rule to estimate which label is more likely.</li>
    <li>It returns a probability-style confidence (when available).</li>
  </ul>

  <div class="callout" style="margin-top: 12px;">
    <strong>Think of it like:</strong> “Based on examples I’ve seen, this looks more similar to group A than group B.”
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
<div class="card">
  <h2>Why the model is intentionally simple</h2>
  <p>
    For a student project focused on awareness and education, a simpler model is a strength:
    it’s easier to explain, easier to debug, and easier to improve as the dataset grows.
  </p>
  <ul>
    <li><b>Transparency:</b> the logic is easier to communicate to non-technical users.</li>
    <li><b>Stability:</b> fewer “mystery behaviors” compared to complex black-box models.</li>
    <li><b>Better learning:</b> the project highlights the full pipeline (data → model → decision).</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
<div class="card">
  <h2>What the model is <span style="color: rgba(239,68,68,0.85);">not</span> doing</h2>
  <ul>
    <li>It is <b>not</b> measuring real fish populations in real time.</li>
    <li>It is <b>not</b> using satellite data, weather, seasons, or ecosystem surveys.</li>
    <li>It is <b>not</b> enforcing laws or issuing official sustainability decisions.</li>
  </ul>
  <p class="muted">
    It’s a learning tool that encourages better thinking — not a replacement for marine science or policy.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
<div class="card">
  <h2>What affects accuracy most</h2>
  <ul>
    <li><b>Label quality:</b> if labels are inconsistent, the model learns inconsistency.</li>
    <li><b>Coverage:</b> if certain species/areas aren’t represented, predictions there are weaker.</li>
    <li><b>Data realism:</b> unrealistic entries lead to meaningless predictions.</li>
  </ul>
  <div class="callout" style="margin-top: 12px;">
    <strong>Best practice:</strong> keep the dataset Bahrain-relevant, consistently labeled, and updated.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
<div class="card">
  <h2>Ethics & responsible use</h2>
  <p>
    Sustainability is high-stakes. Even a simple prediction can influence decisions — so the app treats
    outputs as <b>guidance for awareness</b>, not authority.
  </p>
  <ul>
    <li><b>Don’t over-trust the output:</b> treat it as a starting point for discussion.</li>
    <li><b>Avoid false certainty:</b> a “confident” prediction can still be wrong if data is limited.</li>
    <li><b>Be fair:</b> don’t use the model to judge communities or individuals — use it to improve practices.</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )

st.write("")
st.markdown("</div>", unsafe_allow_html=True)  # end container for columns content

# =========================
# Expanders: deeper (still readable)
# =========================
st.markdown('<div class="container">', unsafe_allow_html=True)

with st.expander("📦 What happens to the data before training (simple but accurate)", expanded=False):
    st.markdown(
        """
<div class="card">
  <h2>Data preparation in this project</h2>
  <ul>
    <li><b>Missing values:</b> filled using common-sense defaults (most frequent for categories, median for numbers).</li>
    <li><b>Categorical inputs:</b> converted using one-hot encoding (turns categories into “on/off” columns).</li>
    <li><b>Numeric input (catch size):</b> kept as a number so the model can learn that higher/lower catch relates to outcomes.</li>
  </ul>
  <p class="muted">
    This is a standard “clean pipeline” approach: it prevents crashes and keeps behavior consistent.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

with st.expander("📊 How model performance is checked (what accuracy means)", expanded=False):
    st.markdown(
        """
<div class="card">
  <h2>Performance checks (short and honest)</h2>
  <p>
    The dataset is split into a training portion and a smaller test portion.
    After training, the model is evaluated on the test set (data it did not train on).
  </p>
  <ul>
    <li><b>Accuracy</b> = how often the model predicts the correct label in the test sample.</li>
    <li>Accuracy is a <b>quick sanity check</b>, not a guarantee of real-world reliability.</li>
  </ul>
  <div class="callout" style="margin-top: 12px;">
    <strong>Reality check:</strong> real sustainability depends on seasons, habitats, regulation, enforcement, and ecosystem health.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

with st.expander("🧭 Limitations (read this like a safety label)", expanded=True):
    st.markdown(
        """
<div class="card">
  <h2>Limitations</h2>
  <ul>
    <li><b>Dataset-limited:</b> the model can’t learn what the dataset doesn’t include.</li>
    <li><b>Label-dependent:</b> the model follows the labeling logic; if the label rules are unclear, predictions suffer.</li>
    <li><b>Context-blind:</b> it doesn’t know time of year, exact location details, habitat maps, or live stock data.</li>
    <li><b>Not policy:</b> it doesn’t represent Bahrain’s official laws or scientific assessments.</li>
  </ul>
  <p class="muted">
    The goal is educational value: understanding trade-offs and encouraging better practice — not issuing final decisions.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

with st.expander("🚀 What would make this stronger next (without repeating the Home page)", expanded=False):
    st.markdown(
        """
<div class="card">
  <h2>Next upgrades that meaningfully improve quality</h2>
  <ul>
    <li><b>Clear labeling guide:</b> a written definition for “sustainable” vs “unsustainable” used consistently.</li>
    <li><b>More Bahrain-specific examples:</b> more scenarios across seasons, areas, and gear types.</li>
    <li><b>Explainability:</b> show which inputs influenced the prediction (e.g., top factors) in a simple way.</li>
    <li><b>Validation:</b> compare against expert feedback or published guidelines (when available).</li>
  </ul>
  <p class="muted">This is the path from “student model” → “strong educational tool.”</p>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
