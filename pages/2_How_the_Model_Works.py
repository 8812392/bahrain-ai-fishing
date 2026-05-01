import streamlit as st

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="How the Model Works",
    page_icon="🤖",
    layout="wide",
)

# =========================
# Professional Ocean Theme
# =========================
OCEAN_CSS = """
<style>
html, body, [class*="css"] {
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}

.stApp {
    background: linear-gradient(180deg, #F7FCFF 0%, #FFFFFF 40%, #F2FBFF 100%);
    color: #0B2233;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    padding-top: 1.2rem;
}

.container {
    max-width: 1080px;
    margin: 0 auto;
}

.hero, .card {
    background: rgba(255,255,255,0.95);
    border: 1px solid rgba(2,132,199,0.14);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 20px rgba(11,34,51,0.05);
}

.hero h1 {
    font-size: 40px;
    margin-bottom: 10px;
}

.hero p, .card p, .card li {
    line-height: 1.6;
    color: rgba(11,34,51,0.82);
}

.badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 14px;
}

.badge {
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(56,189,248,0.10);
    border: 1px solid rgba(2,132,199,0.20);
    font-size: 13px;
}

.flow {
    display: grid;
    grid-template-columns: 1fr auto 1fr auto 1fr;
    gap: 12px;
    align-items: stretch;
}

.flowbox {
    border-radius: 14px;
    padding: 14px;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(2,132,199,0.18);
}

.arrow {
    font-size: 22px;
    font-weight: bold;
    color: rgba(2,132,199,0.75);
    align-self: center;
}

.callout {
    margin-top: 14px;
    padding: 14px;
    border-radius: 14px;
    background: #FFFFFF;
    border: 1px solid rgba(2,132,199,0.18);
}

.muted {
    color: rgba(11,34,51,0.62);
    font-size: 13.5px;
}
</style>
"""
st.markdown(OCEAN_CSS, unsafe_allow_html=True)

# =========================
# Hero Section
# =========================
st.markdown('<div class="container">', unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>How the Model Works</h1>
    <p>
        This page provides a clear overview of how the model functions,
        what informs its predictions, and its practical limitations.
        The focus is on transparency, trust, and responsible use.
    </p>
    <div class="badges">
        <span class="badge">🧠 Explainable Design</span>
        <span class="badge">🧼 Structured Data Pipeline</span>
        <span class="badge">🧭 Ethical Application</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# =========================
# Navigation
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.markdown("**On this page**")
c2.markdown("• System overview")
c3.markdown("• Training & prediction")
c4.markdown("• Limitations & ethics")

st.write("")

# =========================
# Workflow Overview
# =========================
st.markdown("""
<div class="card">
    <h2>Model Workflow Overview</h2>
    <div class="flow">
        <div class="flowbox">
            <h3>1. Scenario Input</h3>
            <p>User provides fishing-related variables such as method, gear, species, and catch size.</p>
        </div>
        <div class="arrow">➜</div>
        <div class="flowbox">
            <h3>2. Data Processing</h3>
            <p>Inputs are cleaned, missing values are handled, and categorical data is transformed for model compatibility.</p>
        </div>
        <div class="arrow">➜</div>
        <div class="flowbox">
            <h3>3. Prediction Output</h3>
            <p>The model estimates whether the scenario is more likely sustainable or unsustainable based on learned patterns.</p>
        </div>
    </div>
    <div class="callout">
        <strong>Important:</strong> This system provides data-driven estimates for educational purposes and should not be treated as an official scientific or regulatory authority.
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# =========================
# Main Sections
# =========================
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown("""
    <div class="card">
        <h2>Training Data Foundation</h2>
        <p>
            The model is trained using structured examples of fishing scenarios,
            where each entry contains operational factors and a sustainability label.
        </p>
        <ul>
            <li><b>Inputs:</b> fishing method, target species, location type, gear, enforcement, catch size.</li>
            <li><b>Outputs:</b> sustainability classification based on example labels.</li>
        </ul>
        <p class="muted">
            Prediction quality depends heavily on dataset completeness, realism, and consistency.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="card">
        <h2>Prediction Logic</h2>
        <p>
            The model identifies statistical relationships between input combinations and historical labels.
            It does not possess ecological understanding; it recognizes learned patterns.
        </p>
        <ul>
            <li>Converts categorical variables into machine-readable formats.</li>
            <li>Applies statistical classification methods.</li>
            <li>Produces probability-style estimates where available.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="card">
        <h2>Why Simplicity Matters</h2>
        <p>
            A simpler model improves explainability, reliability, and educational value,
            making it especially suitable for academic and awareness-focused projects.
        </p>
        <ul>
            <li><b>Transparency:</b> Easier for users to understand.</li>
            <li><b>Stability:</b> Reduced unpredictability.</li>
            <li><b>Scalability:</b> Easier to refine as data expands.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="card">
        <h2>Current Limitations</h2>
        <ul>
            <li>Does not track real-time marine ecosystems.</li>
            <li>Does not use satellite, weather, or biological survey data.</li>
            <li>Does not replace scientific or policy-based sustainability assessments.</li>
        </ul>
        <p class="muted">
            This tool supports awareness and learning rather than direct regulation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="card">
        <h2>Accuracy Factors</h2>
        <ul>
            <li><b>Label consistency</b></li>
            <li><b>Dataset diversity</b></li>
            <li><b>Realistic scenario coverage</b></li>
        </ul>
        <div class="callout">
            <strong>Best Practice:</strong> Maintain Bahrain-specific, updated, and well-labeled data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="card">
        <h2>Ethical Use</h2>
        <p>
            Model outputs should guide awareness and discussion,
            not serve as definitive sustainability judgments.
        </p>
        <ul>
            <li>Avoid over-reliance on predictions.</li>
            <li>Recognize uncertainty.</li>
            <li>Use responsibly to improve practices, not unfairly judge individuals.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
