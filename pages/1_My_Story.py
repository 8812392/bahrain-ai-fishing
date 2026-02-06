import streamlit as st

st.set_page_config(page_title="My Story", layout="centered")

st.title("My Story")

st.markdown(
    """
    This project is personal to me.

    Fishing in Bahrain is more than a tradition — it’s part of our identity.
    Growing up around the sea made me curious about how fishing practices
    affect marine life and the future of our waters.
    """
)

# Centered image using columns (best balance of size + responsiveness)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image(
        "images/IMG_3395.JPG",
        use_container_width=True,
        caption="Me — inspired by the sea and sustainable fishing in Bahrain"
    )

st.markdown(
    """
    I created this AI-powered tool to raise awareness about **sustainable fishing**
    and encourage responsible practices that protect fish populations and coral reefs.

    By combining technology with environmental responsibility, I hope this project
    helps preserve Bahrain’s marine life for future generations.
    """
)
