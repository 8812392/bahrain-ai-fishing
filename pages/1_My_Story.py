import streamlit as st

st.set_page_config(page_title="My Story", layout="wide")

# ---------- Styling (Bright ocean, premium) ----------
st.markdown(
    """
    <style>
      :root{
        --bg:#f6fbff;
        --text:#0f172a;
        --muted:#475569;
        --card:#ffffff;
        --border:#d9e8f5;
        --shadow:rgba(15,23,42,0.06);
        --blue:#0ea5e9;
        --teal:#14b8a6;
      }

      html, body, [class*="css"] { background-color: var(--bg); color: var(--text); }

      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1100px; }

      h1 { font-size: 3.0rem !important; margin-bottom: 0.35rem; }
      h2 { font-size: 1.7rem !important; margin-top: 1.8rem; color: #0b3c5d; }

      p, li { font-size: 1.10rem !important; line-height: 1.80 !important; color: #1e293b; }

      .hero {
        background: linear-gradient(135deg, rgba(14,165,233,0.13), rgba(20,184,166,0.11));
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 18px 18px;
        box-shadow: 0 10px 28px var(--shadow);
      }

      .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 18px 18px;
        box-shadow: 0 10px 28px var(--shadow);
      }

      .muted { color: var(--muted); font-size: 1.05rem; }

      .highlight {
        background: rgba(14,165,233,0.10);
        border: 1px solid rgba(14,165,233,0.25);
        border-radius: 16px;
        padding: 14px 16px;
        color: #0b3c5d;
      }

      .tag {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: #ffffff;
        margin-right: 8px;
        margin-top: 8px;
        font-size: 0.95rem;
        color: var(--text);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
      <h1>My Story</h1>
      <div class="muted">Why I built this project — and why it matters to Bahrain.</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ---------- Top section: story + image ----------
left, right = st.columns([1.55, 1.0], vertical_alignment="top")

with left:
    st.markdown(
        """
        <div class="card">
          <h2 style="margin-top:0;">A project I genuinely care about</h2>
          <p>
            In Bahrain, fishing isn’t just something people do — it’s part of our culture and daily life.
            For me, fishing became something personal: a place where I feel calm, focused, and connected to the sea.
          </p>
          <p>
            Over time, I started paying attention to how different practices affect marine life. Some methods are
            naturally more selective and responsible, while others increase bycatch, damage habitats, or put long-term
            pressure on fish populations.
          </p>
          <p>
            That’s why I built this project: to combine something I love with technology, and create an educational tool
            that helps people understand the difference between sustainable and unsustainable fishing.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        """
        <div class="highlight">
          <b>My goal:</b> make sustainability clearer and more actionable — so better decisions become easier.
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image("images/IMG_3395.JPG", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- What the project does ----------
st.markdown(
    """
    <h2>What this app does</h2>
    <div class="card">
      <p>
        The app takes a fishing scenario and uses a machine-learning model trained on sample practices to estimate whether
        the scenario is more likely to be sustainable or unsustainable.
      </p>
      <p>
        It then outputs a <b>sustainability score</b> (percentage) to make the result easier to understand, and it explains
        the main factors that usually increase or reduce risk.
      </p>
      <p class="muted">
        This is an educational project — it does not replace scientific assessments or government policy.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Why AI helps / what you learned ----------
st.markdown("<h2>Why I chose AI for this</h2>", unsafe_allow_html=True)
c1, c2 = st.columns(2, vertical_alignment="top")

with c1:
    st.markdown(
        """
        <div class="card">
          <p><b>It makes trade-offs visible.</b></p>
          <ul>
            <li>Some gear types are lower-impact than others.</li>
            <li>Some areas are more sensitive and need extra protection.</li>
            <li>Catch size, bycatch reduction, and enforcement can change the outcome.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="card">
          <p><b>It taught me real skills.</b></p>
          <ul>
            <li>Researching sustainability factors and turning them into a dataset</li>
            <li>Training a model and improving accuracy by expanding data variety</li>
            <li>Building a clean UI that beginners can understand</li>
            <li>Deploying a real website and maintaining it over time</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Transparency / limitations ----------
st.markdown("<h2>Honesty & limitations</h2>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="card">
      <p>
        Any AI model is only as good as the examples it learns from. This tool is designed to educate and start discussion,
        not to make official decisions.
      </p>
      <ul>
        <li><b>Limited training data</b>: more real-world examples would improve reliability.</li>
        <li><b>Context matters</b>: season, species health, and protected zones can change what is sustainable.</li>
        <li><b>Prediction ≠ truth</b>: it estimates likelihood based on patterns in sample data.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Closing ----------
st.markdown("<h2>What I hope people take from this</h2>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="card">
      <p>
        I want this project to make sustainability feel clear and practical. Even small choices — choosing lower-impact methods,
        reducing bycatch, and respecting sensitive areas — can help protect Bahrain’s marine life over time.
      </p>
      <p>
        Bahrain’s sea is part of who we are. This is my way of using technology to protect it.
      </p>

      <div class="tag">🇧🇭 Bahrain</div>
      <div class="tag">🌊 Ocean sustainability</div>
      <div class="tag">🤖 Responsible AI</div>
      <div class="tag">🎣 Fishing practices</div>
    </div>
    """,
    unsafe_allow_html=True,
)
