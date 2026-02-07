import streamlit as st
import os

st.set_page_config(page_title="My Story", page_icon="👤", layout="wide")

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
.header h1 { margin: 0; font-size: 40px; letter-spacing: -0.02em; }
.header p  { margin: 10px 0 0 0; font-size: 16.5px; line-height: 1.65; color: rgba(11,34,51,0.78); }

.card {
  background: rgba(255,255,255,0.92);
  border: 1px solid rgba(2, 132, 199, 0.14);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 10px 24px rgba(11, 34, 51, 0.05);
}

.section-title { font-size: 24px; margin: 0 0 8px 0; letter-spacing: -0.01em; }
.muted { color: rgba(11,34,51,0.62); font-size: 13.5px; }

.profile-wrap {
  display: flex;
  gap: 18px;
  align-items: center;
  flex-wrap: wrap;
}
.profile-img {
  width: 170px;
  max-width: 40vw;
  border-radius: 18px;
  border: 1px solid rgba(2, 132, 199, 0.18);
  box-shadow: 0 18px 30px rgba(11, 34, 51, 0.10);
  overflow: hidden;
}
.profile-img img { width: 100%; height: auto; display: block; }

.kicker {
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
</style>
"""
st.markdown(OCEAN_CSS, unsafe_allow_html=True)

IMG_PATH = os.path.join("images", "IMG_3395.JPG")


st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown(
    """
<div class="header">
  <span class="kicker">🇧🇭 Student project • 🌊 Ocean-first design</span>
  <h1>My Story</h1>
  <p>
    I built this as a Bahrain student project because the sea is part of who we are — our food, our culture,
    and our future. This tool is my way of combining technology with real-world responsibility:
    helping people understand how fishing choices can affect marine life over time.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

col1, col2 = st.columns([0.95, 1.25], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Me — building this project</h2>', unsafe_allow_html=True)

    if os.path.exists(IMG_PATH):
        st.image(IMG_PATH, use_container_width=True)
        st.markdown(
            '<p class="muted" style="margin-top:10px;">If you ever replace this image, portrait orientation works best.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No image found yet. Add your photo to: images/IMG_3395.JPG (exact name).")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown(
        """
<div class="card">
  <h2 class="section-title">What this app does</h2>
  <p>
    This app uses a simple machine-learning model trained on example fishing practices (Bahrain + the Gulf).
    You enter details such as method, gear, enforcement level, and catch size — then the model estimates whether
    that pattern is more likely sustainable or unsustainable.
  </p>

  <h2 class="section-title" style="margin-top: 18px;">Why it matters</h2>
  <p>
    Sustainable fishing is not just about “catching less.” It’s about protecting future stock levels,
    avoiding habitat damage (like coral and seabeds), and reducing unintended bycatch.
    Small changes in gear choice, enforcement, and catch size can make a huge difference.
  </p>

  <h2 class="section-title" style="margin-top: 18px;">A promise of responsibility</h2>
  <p>
    This is an educational tool — not an official scientific authority. My goal is to raise awareness and
    encourage better decisions, and to keep improving the dataset and the model over time.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

st.write("")
st.markdown(
    """
<div class="card">
  <h2 class="section-title">What I’m aiming for next</h2>
  <ul>
    <li>More Bahrain-specific data and clearer labeling guidelines</li>
    <li>Better transparency about why the model predicted a result</li>
    <li>More educational content and visuals on sustainable practices</li>
  </ul>
  <p class="muted">Thank you for taking this seriously — it genuinely matters.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)
