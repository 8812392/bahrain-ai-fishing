import streamlit as st

st.set_page_config(page_title="My Story", layout="wide")

# ---------- Styling (clean + premium) ----------
st.markdown(
    """
    <style>
      /* Wider readable text column */
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1100px; }

      /* Typography */
      h1 { font-size: 3.0rem !important; margin-bottom: 0.4rem; }
      h2 { font-size: 1.7rem !important; margin-top: 1.8rem; }
      p, li { font-size: 1.08rem !important; line-height: 1.75 !important; }

      /* Soft card sections */
      .card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px 18px;
      }
      .muted { opacity: 0.85; }
      .highlight {
        background: rgba(59, 130, 246, 0.13);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 14px;
        padding: 14px 16px;
      }
      .tag {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.15);
        margin-right: 8px;
        margin-top: 8px;
        font-size: 0.95rem;
        opacity: 0.95;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.title("My Story")
st.markdown(
    """
    <div class="muted">
      Why I built this project — and what I learned along the way.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ---------- Top section: strong text + controlled image ----------
left, right = st.columns([1.55, 1.0], vertical_alignment="top")

with left:
    st.markdown(
        """
        <div class="card">
          <h2 style="margin-top:0;">A project I genuinely care about</h2>
          <p>
            In Bahrain, fishing isn’t just an industry — it’s a part of our culture and daily life.
            Being close to the sea made me curious about what happens when fishing pressure increases,
            or when certain gear or locations cause more harm than people realize.
          </p>
          <p>
            I built this AI tool because I wanted to create something practical and easy to understand:
            a simple way to explore whether a fishing practice is more likely to be <b>sustainable</b>
            or <b>unsustainable</b> based on common factors (gear type, area sensitivity, bycatch reduction,
            enforcement level, and catch size).
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        """
        <div class="highlight">
          <b>My goal:</b> raise awareness and encourage responsible fishing decisions —
          while keeping the app educational, transparent, and easy to use.
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    # Keep image visually present but not dominating:
    # use_container_width=True in a narrower column gives a good size.
    st.image(
        "images/IMG_3395.JPG",
        use_container_width=True,
        caption="Me — building this as a Bahrain student project",
    )
    st.markdown(
        """
        <div class="muted" style="margin-top:10px;">
          Tip: if you ever change the image, keep it portrait-oriented for best fit.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- What the project does ----------
st.markdown(
    """
    <h2>What this app does (in plain language)</h2>
    <div class="card">
      <p>
        The app takes a few fishing details and uses a machine-learning model trained on example practices
        from Bahrain and the Gulf. It then predicts whether the combination of choices looks more like
        a sustainable practice or an unsustainable one.
      </p>
      <p class="muted">
        This is an educational project — it does not replace scientific assessments or government policy.
        But it can still help people understand which factors usually increase risk.
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
          <p><b>1) It makes trade-offs visible.</b></p>
          <ul>
            <li>Some gear types are lower-impact than others.</li>
            <li>Some areas are more sensitive and need extra protection.</li>
            <li>Catch size and enforcement can change the sustainability outcome.</li>
          </ul>
          <p class="muted">
            AI helps combine these factors into one clear prediction people can discuss.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="card">
          <p><b>2) It taught me real skills.</b></p>
          <ul>
            <li>Data cleaning and preparing features for training</li>
            <li>Training and evaluating a model (and understanding accuracy limits)</li>
            <li>Building a clean user experience with Streamlit</li>
            <li>Deploying and maintaining a live web app</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Transparency / limitations (important for “serious” feel) ----------
st.markdown("<h2>Honesty & limitations</h2>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="card">
      <p>
        Any AI model is only as good as the examples it learns from.
        This classifier is meant to educate and support discussion — not to make official decisions.
      </p>
      <ul>
        <li><b>Training data is limited</b>: more real-world observations would improve reliability.</li>
        <li><b>Context matters</b>: seasonality, protected zones, and species health can change outcomes.</li>
        <li><b>Prediction ≠ truth</b>: it estimates likelihood based on patterns in sample data.</li>
      </ul>
      <p class="muted">
        I included this section because transparency is part of responsible AI.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Closing (strong finish) ----------
st.markdown("<h2>What I hope people take from this</h2>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="card">
      <p>
        I want this project to make sustainability feel <b>clear</b> and <b>actionable</b>.
        Even small changes — choosing lower-impact gear, reducing bycatch, respecting sensitive areas —
        can help protect fish stocks and coral reefs over time.
      </p>
      <p>
        Bahrain’s sea is part of who we are. This is my way of using technology to protect it.
      </p>

      <div class="tag">🇧🇭 Bahrain</div>
      <div class="tag">🌊 Marine life</div>
      <div class="tag">🤖 Responsible AI</div>
      <div class="tag">🎣 Sustainable fishing</div>
    </div>
    """,
    unsafe_allow_html=True,
)
