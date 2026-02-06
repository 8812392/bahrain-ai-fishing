import streamlit as st

st.set_page_config(
    page_title="My Story",
    page_icon="🎣",
    layout="wide"
)

st.markdown("""
<style>
.story-card {
    background: white;
    border-radius: 18px;
    padding: 36px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='story-card'>", unsafe_allow_html=True)

st.title("My Story 🎣")

st.image(
    "images/IMG_3395.JPG",
    caption="Fishing has always been part of my life in Bahrain.",
    use_container_width=True
)

st.write("""
I built this project as a student in Bahrain with a personal connection to the sea.
Fishing is deeply tied to our culture, food, and daily life — but it also comes with
responsibility.

As I learned more about environmental science and technology, I began to see how
data and simple AI-driven tools could help people **understand sustainability in a
practical way**, not just as a theory.

This application is not meant to replace experts or regulations. Instead, it aims to:
- Raise awareness
- Encourage responsible decision-making
- Show how students can use technology for real-world impact

Protecting marine life is not just a global issue — it is personal to me, my community,
and Bahrain’s future.
""")

st.markdown("</div>", unsafe_allow_html=True)
