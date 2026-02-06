import streamlit as st

st.set_page_config(page_title="My Story", page_icon="👤", layout="wide")

st.markdown(
    """
<style>
.stApp {
    background: linear-gradient(180deg, #f6fcff 0%, #e9f7ff 100%);
    color: #0f2a44;
    font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}
h1, h2, h3 {
    color: #0f3d5e;
    font-weight: 800;
}
p, li {
    font-size: 1.08rem !important;
    line-height: 1.78 !important;
    color: #123a57 !important;
}
.card {
    background: #ffffff;
    padding: 1.8rem;
    border-radius: 18px;
    margin: 0.8rem 0 1.2rem 0;
    box-shadow: 0 10px 28px rgba(15, 61, 94, 0.08);
    border: 1px solid rgba(34, 122, 173, 0.10);
    border-top: 5px solid #46b3e6;
}
.muted {
    color: rgba(18,58,87,0.75) !important;
    font-size: 0.98rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("<h1>My Story</h1>", unsafe_allow_html=True)

left, right = st.columns([1.2, 1.0], gap="large")

with left:
    st.markdown(
        """
<div class="card">
<p>
I built this project as a Bahrain student because the sea is part of our identity — our coastline, our traditions,
and the communities that depend on it.
</p>

<p>
Over time, fishing pressure can grow quietly: higher catches, wider effort, and methods that unintentionally impact
young fish, coral reefs, or already-stressed species. I wanted to create something that’s simple, interactive,
and understandable — a tool that encourages people to think about “how” we fish, not just “what” we catch.
</p>

<p>
This app is not meant to replace science or regulation. It’s an educational project that uses a small machine-learning
model trained on example practices. The goal is awareness: showing that choices like gear, location, enforcement, and
catch size can change outcomes.
</p>

<ul>
  <li><b>Purpose:</b> raise awareness about responsible fishing</li>
  <li><b>Focus:</b> Bahrain and Gulf-style fishing scenarios</li>
  <li><b>Goal:</b> support sustainability conversations in a clear way</li>
</ul>

<p class="muted" style="margin-top: 0.9rem;">
If I update the dataset or the model in the future, the app can become even more accurate — but the mission stays the same:
protect marine life and keep fishing sustainable for the next generations.
</p>
</div>
""",
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
<div class="card">
<h3 style="margin-top:0;">Me — building this as a Bahrain student project</h3>
<p class="muted" style="margin-top:-0.2rem;">
This photo is used inside the app as part of the story behind why I built it.
</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Your uploaded image path (make sure it exists in repo)
    st.image(
        "images/IMG_3395.JPG",
        caption="Bahrain – inspiration behind the project",
        width=360,
    )

    st.markdown(
        """
<div class="card">
<h3 style="margin-top:0;">What this app does</h3>
<ul>
  <li>Takes fishing details (method, location type, enforcement level, etc.)</li>
  <li>Uses a trained model to estimate sustainable vs unsustainable</li>
  <li>Displays a clear prediction as an educational guide</li>
</ul>
</div>
""",
        unsafe_allow_html=True,
    )
