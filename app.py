import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ---------------- Page config ---------------- #
st.set_page_config(
    page_title="Bahrain Sustainable Fishing Classifier",
    page_icon="🇧🇭",
    layout="centered",
)

# ---------------- Title & Introduction ---------------- #

st.title("AI Tool for Sustainable Fishing in Bahrain")

st.write("""
This website uses **artificial intelligence** to analyze fishing practices
and predict whether they are **sustainable** or **unsustainable**.

The goal of this project is to raise awareness about responsible fishing
and protect marine life in Bahrain.
""")

st.subheader("How to use this website")

st.write("""
1. Select the fishing method  
2. Enter the fishing details  
3. Click **Predict** to see the result
""")

st.info(
    "Sustainable fishing helps protect fish populations, coral reefs, "
    "and the future of fishing for the next generations."
)

# ---------------- Load data + model ---------------- #

@st.cache_resource
def load_model_and_metadata():
    # Load your cleaned CSV (no underscores, with new columns)
    data = pd.read_csv("bahrain_fishing_practices.csv")

    # Features and label
    X = data.drop(columns=["sustainable_practice"])
    y = data["sustainable_practice"]

    numeric_features = ["catch_per_trip_kg", "gear_impact_score", "depth_m"]
    categorical_features = [
        "method",
        "target_species",
        "area_type",
        "uses_bycatch_reduction",
        "gear_type",
        "enforcement_level",
        "target_status",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        max_depth=None,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", rf),
        ]
    )

    # Train / test split so we can show accuracy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    # Collect unique values for dropdowns (so they always match the CSV)
    metadata = {
        "methods": sorted(data["method"].unique()),
        "species": sorted(data["target_species"].unique()),
        "areas": sorted(data["area_type"].unique()),
        "uses_bycatch": sorted(data["uses_bycatch_reduction"].unique()),
        "gears": sorted(data["gear_type"].unique()),
        "enforcement": sorted(data["enforcement_level"].unique()),
        "statuses": sorted(data["target_status"].unique()),
    }

    return model, test_accuracy, metadata


model, test_accuracy, meta = load_model_and_metadata()

# Helper: compute gear impact and depth automatically from choices
def estimate_gear_impact(gear_type: str) -> int:
    gear_type = gear_type.lower()
    if "line" in gear_type:
        return 2
    if "hand" in gear_type:
        return 2
    if "trap" in gear_type:
        return 4
    if "drift" in gear_type:
        return 8
    if "purse" in gear_type:
        return 6
    if "gill" in gear_type or "net" in gear_type or "trawl" in gear_type:
        return 9
    if "cage" in gear_type:
        return 3
    return 5  # default medium


def estimate_depth(area_type: str) -> int:
    area_type = area_type.lower()
    if "nearshore" in area_type:
        return 30
    if "reef" in area_type:
        return 20
    if "offshore" in area_type:
        return 120
    if "deep" in area_type:
        return 300
    return 100  # default


# ---------------- UI: Header ---------------- #

st.markdown(
    """
    # 🇧🇭 Bahrain Sustainable Fishing Classifier

    This app uses a simple **machine-learning model** trained on example fishing
    practices from Bahrain and the Gulf to guess whether a practice is
    **sustainable** or **unsustainable**.

    👉 Fill in the details below, then click **“Check sustainability”** to see the prediction.

    _This is an educational project, not an official scientific tool._
    """
)

st.info(f"**Current test accuracy:** about **{test_accuracy * 100:.1f}%** on held-out data.")

st.divider()

# ---------------- UI: Input form ---------------- #

st.subheader("Step 1 – Describe the fishing practice")

col1, col2 = st.columns(2)

with col1:
    method = st.selectbox(
        "Fishing method",
        meta["methods"],
        help="How the fish are caught (handline, trawling, purse seine, etc.).",
    )

    area_type = st.selectbox(
        "Area type",
        meta["areas"],
        help="Where the fishing happens (nearshore, offshore, reef area, etc.).",
    )

    uses_bycatch = st.selectbox(
        "Uses bycatch reduction devices?",
        meta["uses_bycatch"],
        help="Devices that reduce accidental catch of turtles, dolphins, seabirds, etc.",
    )

with col2:
    species = st.selectbox(
        "Target species",
        meta["species"],
        help="Main species being targeted on this trip.",
    )

    gear_type = st.selectbox(
        "Gear type",
        meta["gears"],
        help="Type of gear used (net, line, trap, cage, etc.).",
    )

    enforcement = st.selectbox(
        "Enforcement level",
        meta["enforcement"],
        help="How strongly fishing rules are checked and enforced in that area.",
    )

catch = st.slider(
    "Catch per trip (kg)",
    min_value=20,
    max_value=1000,
    value=300,
    step=10,
    help="Approximate amount of fish caught in a single trip.",
)

status = st.selectbox(
    "Status of target species",
    meta["statuses"],
    help="Is the species abundant or under pressure (vulnerable)?",
)

st.caption(
    "In general, very high catch combined with vulnerable species, high-impact gear and sensitive areas "
    "is more likely to be unsustainable."
)

st.divider()

# ---------------- Prediction ---------------- #

st.subheader("Step 2 – Check sustainability")

if st.button("✅ Check sustainability"):
    # Automatically estimate gear impact score and depth
    gear_impact = estimate_gear_impact(gear_type)
    depth_m = estimate_depth(area_type)

    input_df = pd.DataFrame(
        [
            {
                "method": method,
                "target_species": species,
                "catch_per_trip_kg": catch,
                "area_type": area_type,
                "uses_bycatch_reduction": uses_bycatch,
                "gear_type": gear_type,
                "enforcement_level": enforcement,
                "target_status": status,
                "gear_impact_score": gear_impact,
                "depth_m": depth_m,
            }
        ]
    )

    proba = model.predict_proba(input_df)[0]  # [P(class 0), P(class 1)]
    p_unsust = float(proba[0]) * 100
    p_sust = float(proba[1]) * 100

    pred_class = int(proba[1] >= 0.5)

    if pred_class == 1:
        st.success(
            f"🌱 The model predicts this practice is **SUSTAINABLE** "
            f"(about **{p_sust:.1f}%** confidence)."
        )
        st.write(
            "Based on similar examples in the training data, this combination of method, "
            "catch size, gear, area and species usually appears to put **limited pressure** "
            "on Bahrain’s marine ecosystem."
        )
    else:
        st.error(
            f"⚠️ The model predicts this practice is **UNSUSTAINABLE** "
            f"(about **{p_unsust:.1f}%** confidence)."
        )
        st.write(
            "High catch, vulnerable species, high-impact gear, sensitive areas or weak enforcement "
            "often make a practice unsustainable. According to the patterns it has learned, "
            "this trip could put **too much pressure** on marine life."
        )

    st.caption("Percentages show how confident the model is, based on patterns in the dataset you created.")

# ---------------- Explanation ---------------- #

with st.expander("How does this AI work? (simple explanation)"):
    st.markdown(
        """
        - You collected example fishing practices around Bahrain and labeled each one as
          **sustainable (1)** or **unsustainable (0)**.  
        - The model looks at patterns in:
          - fishing method and gear type  
          - catch per trip (kg)  
          - area (nearshore / offshore / reef) and estimated depth  
          - whether bycatch reduction devices are used  
          - enforcement strength  
          - species status (abundant vs vulnerable)  
          - gear impact score (0–10)  
        - It splits the data into **training** (80%) and **test** (20%) examples.  
        - Accuracy is measured on the **test** part only, so it reflects how well the model
          generalizes to new, unseen trips.

        This project doesn’t replace scientific stock assessments, but it’s a useful way to
        explore how different choices in fishing can affect sustainability in Bahrain.
        """
    )

st.markdown("---")
st.caption("Project by a Bahrain student using Python, scikit-learn and Streamlit 🌊🎣")
