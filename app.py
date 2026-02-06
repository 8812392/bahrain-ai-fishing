import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Bahrain Sustainable Fishing Classifier",
    page_icon="🌊",
    layout="wide",
)


# -----------------------------
# OCEAN THEME (LIGHT, PREMIUM)
# -----------------------------
st.markdown(
    """
<style>
/* Global background */
.stApp {
    background: linear-gradient(180deg, #f6fcff 0%, #e9f7ff 100%);
    color: #0f2a44;
    font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}

/* Headings */
h1, h2, h3 {
    color: #0f3d5e;
    font-weight: 800;
    letter-spacing: -0.02em;
}

/* Body text */
p, li, label {
    font-size: 1.05rem !important;
    line-height: 1.75 !important;
    color: #123a57 !important;
}

/* Card container */
.card {
    background: #ffffff;
    padding: 1.6rem 1.6rem;
    border-radius: 18px;
    margin: 0.8rem 0 1.2rem 0;
    box-shadow: 0 10px 28px rgba(15, 61, 94, 0.08);
    border: 1px solid rgba(34, 122, 173, 0.10);
}

/* Accent top border */
.card-accent {
    border-top: 5px solid #46b3e6;
}

/* Smaller muted text */
.muted {
    color: rgba(18,58,87,0.75) !important;
    font-size: 0.98rem !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #e9f7ff !important;
    border-right: 1px solid rgba(34, 122, 173, 0.12);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #46b3e6, #2aa7cf) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.15rem !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 10px 20px rgba(42,167,207,0.18) !important;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #2aa7cf, #1a92c2) !important;
    transform: translateY(-1px);
}

/* Inputs */
div[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input {
    background: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid rgba(34, 122, 173, 0.18) !important;
}

/* Sliders */
div[role="slider"] {
    color: #2aa7cf !important;
}

/* Metric blocks */
div[data-testid="stMetric"] {
    background: white;
    border-radius: 14px;
    padding: 0.8rem;
    border: 1px solid rgba(34, 122, 173, 0.10);
    box-shadow: 0 8px 22px rgba(15, 61, 94, 0.06);
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# HELPERS
# -----------------------------
def card(title: str, body_html: str, accent: bool = True):
    cls = "card card-accent" if accent else "card"
    st.markdown(
        f"""
<div class="{cls}">
  <h3 style="margin: 0 0 0.6rem 0;">{title}</h3>
  {body_html}
</div>
""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    """
    Trains a simple, robust baseline model.
    Detects columns based on common names.
    """
    # Try to detect target column (label)
    target_col = pick_first_existing_column(
        df,
        [
            "label",
            "sustainability",
            "sustainable",
            "is_sustainable",
            "result",
            "target",
            "y",
            "class",
        ],
    )

    if target_col is None:
        return None, None, None, None, (
            "Couldn't find the target/label column in your CSV.\n\n"
            "Expected something like: label, sustainable, sustainability, is_sustainable, result."
        )

    # Detect feature columns (common names)
    feature_map = {
        "fishing_method": ["fishing_method", "method", "gear_method", "fishing type"],
        "target_species": ["target_species", "species", "target", "fish_species"],
        "area_type": ["area_type", "area", "zone", "location_type"],
        "gear_type": ["gear_type", "gear", "tool"],
        "bycatch_reduction": ["bycatch_reduction", "bycatch_device", "uses_bycatch_devices", "bycatch"],
        "enforcement_level": ["enforcement_level", "enforcement", "regulation", "compliance"],
        "catch_per_trip_kg": ["catch_per_trip_kg", "catch_per_trip", "catch_kg", "catch"],
        "target_status": ["target_status", "status", "stock_status"],
    }

    resolved = {}
    for key, candidates in feature_map.items():
        col = pick_first_existing_column(df, candidates)
        if col is not None:
            resolved[key] = col

    # Require a minimum set (so model makes sense)
    required_keys = ["fishing_method", "target_species", "area_type", "gear_type", "enforcement_level"]
    missing_required = [k for k in required_keys if k not in resolved]

    if missing_required:
        return None, None, None, None, (
            "Your CSV is missing some expected feature columns.\n\n"
            f"Missing: {missing_required}\n\n"
            f"Found columns: {list(df.columns)}"
        )

    # Build X/y
    used_feature_cols = list(resolved.values())
    work = df[used_feature_cols + [target_col]].dropna().copy()

    X = work[used_feature_cols]
    y = work[target_col]

    # If y is strings, keep it; logistic regression can handle with encoding via label? sklearn expects 1d. We'll keep.
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    # Identify types
    cat_cols = []
    num_cols = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    clf = LogisticRegression(max_iter=2000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Build "choices" lists for UI
    choices = {}
    for key, col in resolved.items():
        if col in cat_cols:
            vals = sorted(work[col].astype(str).unique().tolist())
            choices[key] = vals
        else:
            # numeric
            choices[key] = {
                "min": float(np.nanmin(work[col])),
                "max": float(np.nanmax(work[col])),
                "median": float(np.nanmedian(work[col])),
            }

    return pipe, resolved, choices, acc, None


# -----------------------------
# LOAD + TRAIN
# -----------------------------
CSV_PATH = "bahrain_fishing_practices.csv"
if not os.path.exists(CSV_PATH):
    st.error(f"Can't find `{CSV_PATH}` in the repo root. Upload it next to `app.py`.")
    st.stop()

df = load_data(CSV_PATH)
model, resolved_cols, choices, test_acc, train_error = train_model(df)

# -----------------------------
# HERO
# -----------------------------
left, right = st.columns([1.4, 1.0], gap="large")

with left:
    st.markdown("<h1>AI Tool for Sustainable Fishing in Bahrain</h1>", unsafe_allow_html=True)
    st.markdown(
        """
<p class="muted" style="margin-top: -0.2rem;">
This app uses a simple machine-learning model trained on example fishing practices from Bahrain and the Gulf
to estimate whether a practice is more likely <b>sustainable</b> or <b>unsustainable</b>.
</p>
""",
        unsafe_allow_html=True,
    )

with right:
    if test_acc is not None:
        st.metric("Current test accuracy", f"{test_acc*100:.1f}%")
    else:
        st.metric("Current test accuracy", "—")

# -----------------------------
# INTRO + HOW TO USE
# -----------------------------
card(
    "How to use this website",
    """
<ol style="margin: 0.2rem 0 0 1.1rem;">
  <li>Select the fishing method and details below</li>
  <li>Click <b>Check sustainability</b></li>
  <li>Use the result as an educational guide</li>
</ol>
<p class="muted" style="margin-top: 0.9rem;">
This is an educational project, not an official scientific tool.
</p>
""",
)

# If training had an error, show it clearly and stop
if train_error:
    card(
        "Setup issue (CSV columns)",
        f"""
<p>{train_error.replace("\n","<br>")}</p>
<p class="muted" style="margin-top:0.8rem;">
If you want, send me a screenshot of your CSV column headers (top row) and I’ll align the app perfectly to your dataset.
</p>
""",
        accent=False,
    )
    st.stop()

# -----------------------------
# CLASSIFIER UI
# -----------------------------
card(
    "Step 1 — Describe the fishing practice",
    "<p class='muted' style='margin-top:-0.3rem;'>Choose realistic values to get the most meaningful prediction.</p>",
)

c1, c2 = st.columns(2, gap="large")

with c1:
    fishing_method = st.selectbox("Fishing method", choices["fishing_method"], index=0)
    area_type = st.selectbox("Area type", choices["area_type"], index=0)
    bycatch = None
    if "bycatch_reduction" in choices and isinstance(choices["bycatch_reduction"], list):
        bycatch = st.selectbox("Uses bycatch reduction devices?", choices["bycatch_reduction"], index=0)

with c2:
    target_species = st.selectbox("Target species", choices["target_species"], index=0)
    gear_type = st.selectbox("Gear type", choices["gear_type"], index=0)
    enforcement_level = st.selectbox("Enforcement level", choices["enforcement_level"], index=0)

catch_per_trip = None
if "catch_per_trip_kg" in choices and isinstance(choices["catch_per_trip_kg"], dict):
    rng = choices["catch_per_trip_kg"]
    # make slider nice (and not extreme)
    low = max(0.0, rng["min"])
    high = max(low + 1.0, rng["max"])
    default = float(np.clip(rng["median"], low, high))
    catch_per_trip = st.slider("Catch per trip (kg)", float(low), float(high), float(default))

target_status = None
if "target_status" in choices and isinstance(choices["target_status"], list):
    target_status = st.selectbox("Status of target species", choices["target_status"], index=0)

card(
    "Step 2 — Check sustainability",
    "<p class='muted' style='margin-top:-0.3rem;'>The prediction is based on patterns in the example dataset used for training.</p>",
)

# Assemble input row based on resolved columns
def build_input_row():
    row = {}
    # required
    row[resolved_cols["fishing_method"]] = fishing_method
    row[resolved_cols["target_species"]] = target_species
    row[resolved_cols["area_type"]] = area_type
    row[resolved_cols["gear_type"]] = gear_type
    row[resolved_cols["enforcement_level"]] = enforcement_level

    # optional
    if "bycatch_reduction" in resolved_cols and bycatch is not None:
        row[resolved_cols["bycatch_reduction"]] = bycatch
    if "catch_per_trip_kg" in resolved_cols and catch_per_trip is not None:
        row[resolved_cols["catch_per_trip_kg"]] = catch_per_trip
    if "target_status" in resolved_cols and target_status is not None:
        row[resolved_cols["target_status"]] = target_status

    return pd.DataFrame([row])


btn = st.button("✅ Check sustainability")

if btn:
    x_in = build_input_row()
    pred = model.predict(x_in)[0]

    proba = None
    try:
        probs = model.predict_proba(x_in)[0]
        # if binary, show max probability as confidence
        proba = float(np.max(probs))
    except Exception:
        proba = None

    # Style result
    if str(pred).strip().lower() in ["sustainable", "1", "true", "yes"]:
        headline = "✅ Likely Sustainable"
        detail = "Based on the training data patterns, this looks more aligned with sustainable practices."
    else:
        headline = "⚠️ Likely Unsustainable"
        detail = "Based on the training data patterns, this looks more aligned with unsustainable practices."

    card(
        headline,
        f"""
<p style="font-size:1.12rem; margin-top:-0.2rem;"><b>Prediction:</b> {pred}</p>
<p>{detail}</p>
{"<p class='muted'><b>Confidence:</b> " + f"{proba*100:.1f}%" + "</p>" if proba is not None else ""}
""",
        accent=True,
    )

with st.expander("How does this AI work?"):
    st.markdown(
        """
- The app learns from examples in the CSV (past fishing practices + a label).
- It converts text choices (like “bottom longline”) into numeric features.
- A simple classifier learns patterns that correlate with sustainable vs unsustainable outcomes.
- Your input is compared to those learned patterns to make a prediction.

**Note:** Accuracy depends heavily on the size and quality of the training dataset.
"""
    )

st.markdown(
    "<div class='muted' style='text-align:center; margin-top:1.2rem;'>Project by a Bahrain student using Python, scikit-learn and Streamlit 🌊</div>",
    unsafe_allow_html=True,
)
