import streamlit as st

# Page configuration
st.set_page_config(
    page_title="My Story",
    page_icon="🎣",
    layout="wide"
)

# ---------- CUSTOM STYLING ----------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 30px;
        font-weight: 600;
        margin-top: 40px;
        margin-bottom: 15px;
    }
    .body-text {
        font-size: 18px;
        line-height: 1.7;
        max-width: 900px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- TITLE ----------
st.markdown('<div class="main-title">My Story</div>', unsafe_allow_html=True)

# ---------- LAYOUT ----------
col1, col2 = st.columns([1, 1.3])

with col1:
    st.image(
        "images/IMG_3395.JPG",
        use_container_width=True
    )

with col2:
    st.markdown(
        """
        <div class="body-text">
        Fishing has always been more than just a hobby for me — it is something deeply connected
        to my childhood, my family, and my environment. Growing up in Bahrain, the sea has always
        been a constant presence in my life. From early mornings by the shore to long conversations
        with experienced fishermen, I learned to respect the ocean and understand how closely our
        lives are tied to it.

        Over time, I began to notice a shift. Fish populations were changing, certain species were
        becoming harder to find, and conversations started to include concerns about overfishing,
        destructive gear, and unsustainable practices. This made me realize that fishing is not
        just about catching fish — it is about responsibility.

        This project represents my attempt to combine something I genuinely love with modern
        technology to create awareness and encourage smarter choices. By using artificial
        intelligence, data, and clear information, I wanted to build something that helps people
        understand the difference between sustainable and unsustainable fishing — not in an
        abstract way, but in a practical, real-world context.
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- WHAT THE APP DOES ----------
st.markdown('<div class="section-title">What this app does</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="body-text">
    This application is designed to educate and assist users in understanding fishing practices
    and their environmental impact. It uses an AI-powered system to analyze fishing methods, gear
    types, species, and locations, and then provides insight into whether those practices are
    sustainable or harmful to marine ecosystems.

    Instead of relying on assumptions, the app focuses on data-driven reasoning. It highlights
    why certain methods damage seabed habitats, increase bycatch, or threaten fish populations,
    while also promoting techniques that allow marine life to recover and remain balanced.

    The goal is not to judge fishermen, but to empower better decision-making — whether the user
    is a student, recreational fisher, or someone simply interested in ocean conservation.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- WHY THIS PROJECT MATTERS ----------
st.markdown('<div class="section-title">Why this project matters</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="body-text">
    Bahrain’s marine environment is fragile and extremely important to our culture, economy, and
    food security. Unsustainable fishing practices may not show immediate consequences, but their
    long-term effects can permanently damage ecosystems.

    By creating this project, I wanted to show how technology can be used as a tool for
    environmental responsibility. This app demonstrates that artificial intelligence is not only
    about automation or convenience — it can also support sustainability, awareness, and future
    planning.

    This project reflects both my personal values and my learning journey. It challenged me to
    research deeply, think critically, and design something meaningful rather than superficial.
    </div>
    """,
    unsafe_allow_html=True
)
