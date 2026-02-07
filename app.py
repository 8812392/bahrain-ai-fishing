import os
import textwrap
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Bahrain Sustainable Fishing Classifier",
    page_icon="🇧🇭",
    layout="wide",
)

# =========================
# Ocean theme CSS
# =========================
OCEAN_CSS = """
<style>
/* --- Global background and typography --- */
html, body, [class*="css"]  {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
}

/* Streamlit app background */
.stApp {
  background: linear-gradient(180deg, #F7FCFF 0%, #FFFFFF 40%, #F2FBFF 100%);
  color: #0B2233;
}

/* Remove default Streamlit footer/menu clutter (keeps Share etc) */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* --- Layout helpers --- */
.container {
  max-width: 1080px;
  margin: 0 auto;
}

/* --- Hero header --- */
.hero {
  background: radial-gradient(1200px 400px at 20% 0%, rgba(24, 163, 200, 0.18), rgba(255,255,255,0) 55%),
              radial-gradient(900px 380px at 100% 20%, rgba(56, 189, 248, 0.14), rgba(255,255,255,0) 50%),
              linear-gradient(180deg, #FFFFFF 0%, #F7FCFF 100%);
  border: 1px solid rgba(13, 148, 136, 0.10);
  border-radius: 20px;
  padding: 22px 22px 18px 22px;
  box-shadow: 0 12px 28px rgba(11, 34, 51, 0.06);
}

.hero h1 {
  font-size: 44px;
  line-height: 1.1;
  margin: 0;
  letter-spacing: -0.02em;
}

.hero p {
  font-size: 16.5px;
  line-height: 1.6;
  margin: 10px 0 0 0;
  color: rgba(11, 34, 51, 0.78);
}

.badges {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 14px;
}

.badge {
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

/* --- Cards --- */
.card {
  background: rgba(255,255,255,0.92);
  border: 1px solid rgba(2, 132, 199, 0.14);
  border-radius: 18px;
  padding: 18px 18px 16px 18px;
  box-shadow: 0 10px 24px rgba(11, 34, 51, 0.05);
}

.card h2 {
  margin: 0 0 10px 0;
  font-size: 26px;
  letter-spacing: -0.01em;
}

.card p, .card li {
  color: rgba(11, 34, 51, 0.82);
  font-size: 15.5px;
  line-height: 1.6;
}

/* --- Section headings --- */
.section-title {
  font-size: 28px;
  margin: 0 0 10px 0;
  letter-spacing: -0.01em;
}

/* --- Buttons --- */
.stButton>button {
  border-radius: 12px;
  border: 1px solid rgba(2, 132, 199, 0.25);
  background: linear-gradient(180deg, #E0F6FF 0%, #CDEFFF 100%);
  color: #06324A;
  padding: 10px 14px;
  font-weight: 650;
  box-shadow: 0 10px 18px rgba(11,34,51,0.06);
}
.stButton>button:hover {
  border: 1px solid rgba(2, 132, 199, 0.35);
  background: linear-gradient(180deg, #D4F3FF 0%, #BFEAFF 100%);
}

/* --- Inputs --- */
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
  border-radius: 12px !important;
}

/* --- Prediction result pill --- */
.pill {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 12px 14px;
  border-radius: 14px;
  border: 1px solid rgba(2,132,199,0.20);
  background: rgba(56, 189, 248, 0.10);
  font-size: 16px;
  font-weight: 700;
  color: #06324A;
}

.pill.good {
  border-color: rgba(16,185,129,0.26);
  background: rgba(16,185,129,0.10);
  color: #064E3B;
}
.pill.bad {
  border-color: rgba(239,68,68,0.22);
  background: rgba(239,68,68,0.08);
  color: #7F1D1D;
}

/* --- Small muted text --- */
.muted {
  color: rgba(11, 34, 51, 0.62);
  font-size: 13.5px;
}

/* Reduce huge top padding a bit */
.block-container {
  padding-top: 1.2rem;
}
</style>
"""
st.markdown(OCEAN_CSS, unsafe_allow_html=True)


# =========================
# Helpers
# =========================
CSV_PATH = "bahrain_fishing_practices.csv"

# These are the "expected" columns the model works with.
# If your CSV uses different names, we map them in normalize_columns().
EXPECTED_FEATURES = [
    "fishing_method",
    "target_species",
    "area_type",
    "gear_type",
    "uses_bycatch_reduction_devices",
    "enforcement_level",
    "catch_per_trip_kg",
    "status_of_target_species",
]
TARGET_COL_CANDIDATES = ["label", "sustainability", "is_sustainable", "target", "class"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make common column naming more forgiving (case, spaces, etc).
    You can extend this mapping if your CSV uses different names.
    """
    rename_map = {}

    # Lowercase, strip
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    # Friendly synonyms mapping
    synonyms = {
        "Fishing method": "fishing_method",
        "fishing method": "fishing_method",
        "method": "fishing_method",
        "Target species": "target_species",
        "target species": "target_species",
        "species": "target_species",
        "Area type": "area_type",
        "area type": "area_type",
        "area": "area_type",
        "Gear type": "gear_type",
        "gear type": "gear_type",
        "gear": "gear_type",
        "Uses bycatch reduction devices?": "uses_bycatch_reduction_devices",
        "uses bycatch reduction devices": "uses_bycatch_reduction_devices",
        "bycatch devices": "uses_bycatch_reduction_devices",
        "Enforcement level": "enforcement_level",
        "enforcement level": "enforcement_level",
        "Catch per trip (kg)": "catch_per_trip_kg",
        "catch per trip (kg)": "catch_per_trip_kg",
        "catch_per_trip": "catch_per_trip_kg",
        "Status of target species": "status_of_target_species",
        "status of target species": "status_of_target_species",
        "status": "status_of_target_species",
    }

    for c in df.columns:
        if c in synonyms:
            rename_map[c] = synonyms[c]

    df = df.rename(columns=rename_map)

    # If any expected feature missing but a similar column exists with different casing/underscores:
    # (lightweight normalization)
    norm = {c: c.lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=norm)

    return df


def find_target_column(df: pd.DataFrame) -> str | None:
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in TARGET_COL_CANDIDATES:
        if cand in lower_cols:
            return lower_cols[cand]

    # Also try exact "sustainable" / "unsustainable" label column names people often use
    for c in df.columns:
        if "sustain" in c.lower() and c.lower() not in [*EXPECTED_FEATURES]:
            return c
    return None


def pretty_error(title: str, details: str):
    # FIXED: no backslash inside f-string expression
    safe_details = details.replace("\n", "<br>")
    st.markdown(
        f"""
<div class="container">
  <div class="card">
    <h2>{title}</h2>
    <p>{safe_details}</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    return df


@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame, target_col: str):
    # Validate target labels (we accept a few styles)
    y_raw = df[target_col]

    # Convert target to 0/1 if needed
    def to01(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in ["1", "true", "yes", "sustainable", "sustain", "good"]:
            return 1
        if s in ["0", "false", "no", "unsustainable", "bad"]:
            return 0
        # If already numeric-ish
        try:
            fv = float(s)
            if fv in [0.0, 1.0]:
                return int(fv)
        except Exception:
            pass
        return np.nan

    y = y_raw.map(to01)
    work = df.copy()
    work[target_col] = y
    work = work.dropna(subset=[target_col]).copy()

    # =========================
    # ADJUSTMENT 1: Don't crash on missing feature columns
    # Fill missing columns with safe defaults and warn instead.
    # =========================
    missing = [c for c in EXPECTED_FEATURES if c not in work.columns]
    if missing:
        for m in missing:
            if m == "catch_per_trip_kg":
                work[m] = np.nan
            else:
                work[m] = "unknown"

        st.warning(
            "Your CSV is missing some optional columns, so the app filled them with defaults:\n"
            + "\n".join([f"• {m}" for m in missing])
        )

    X = work[EXPECTED_FEATURES].copy()
    y = work[target_col].astype(int).copy()

    # Ensure numeric
    if "catch_per_trip_kg" in X.columns:
        X["catch_per_trip_kg"] = pd.to_numeric(X["catch_per_trip_kg"], errors="coerce")

    cat_cols = [c for c in EXPECTED_FEATURES if c != "catch_per_trip_kg"]
    num_cols = ["catch_per_trip_kg"]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                num_cols,
            ),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=1000)

    clf = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds) if len(y_test) else None

    # =========================
    # ADJUSTMENT 2: Safer dropdown option lists (won't break if column missing/empty)
    # =========================
    def col_options(name, fallback):
        if name in work.columns:
            vals = sorted(work[name].dropna().astype(str).unique().tolist())
            return vals if vals else fallback
        return fallback

    options = {
        "fishing_method": col_options("fishing_method", ["hand line", "bottom longline", "net"]),
        "target_species": col_options("target_species", ["Crab", "Hamour", "Shrimp"]),
        "area_type": col_options("area_type", ["Nearshore Bahrain", "Offshore Bahrain"]),
        "gear_type": col_options("gear_type", ["line", "net", "trap"]),
        "uses_bycatch_reduction_devices": col_options("uses_bycatch_reduction_devices", ["no", "yes"]),
        "enforcement_level": col_options("enforcement_level", ["low", "medium", "high"]),
        "status_of_target_species": col_options("status_of_target_species", ["healthy", "declining", "overfished"]),
    }

    return clf, acc, options


def clamp_default(options: list[str], preferred: str, fallback_index: int = 0) -> str:
    if preferred in options:
        return preferred
    if options:
        return options[min(fallback_index, len(options) - 1)]
    return preferred


# =========================
# HERO
# =========================
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown(
    """
<div class="hero">
  <h1>AI Tool for Sustainable Fishing in Bahrain 🇧🇭</h1>
  <p>
    An educational Streamlit app that uses a simple machine-learning model trained on example fishing practices
    from Bahrain and the Gulf to predict whether a practice is more likely <b>sustainable</b> or <b>unsustainable</b>.
  </p>
  <div class="badges">
    <span class="badge">🌊 Ocean-themed UI</span>
    <span class="badge">🤖 ML classifier</span>
    <span class="badge">📚 Educational project</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.write("")


# =========================
# Load + Train
# =========================
if not os.path.exists(CSV_PATH):
    pretty_error(
        "Missing dataset file",
        f"I couldn't find <b>{CSV_PATH}</b> in the repository root.<br><br>"
        "Make sure the file exists and is named exactly: <b>bahrain_fishing_practices.csv</b>.",
    )
    st.stop()

try:
    df = load_data(CSV_PATH)
except Exception as e:
    pretty_error("Could not read the CSV", f"{type(e).__name__}: {e}")
    st.stop()

target_col = find_target_column(df)
if target_col is None:
    pretty_error(
        "Dataset label column not found",
        "Your CSV needs a label column that indicates sustainable vs unsustainable.<br><br>"
        "Accepted names include:<br>"
        "• <b>label</b><br>"
        "• <b>sustainability</b><br>"
        "• <b>is_sustainable</b><br><br>"
        "Where values can be like: <b>sustainable/unsustainable</b> or <b>1/0</b>.",
    )
    st.stop()

try:
    clf, acc, options = train_model(df, target_col)
except Exception as e:
    # FIXED: no backslashes inside f-string expression
    msg = f"{type(e).__name__}: {e}"
    pretty_error("Model setup issue", msg)
    st.stop()


# =========================
# Main content (2 columns)
# =========================
st.markdown('<div class="container">', unsafe_allow_html=True)
left, right = st.columns([1.25, 0.95], gap="large")

with left:
    st.markdown(
        """
<div class="card">
  <h2>Step 1 — Describe the fishing practice</h2>
  <p class="muted">Choose realistic values and then run the prediction.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        fishing_method = st.selectbox(
            "Fishing method",
            options.get("fishing_method", ["bottom longline", "handline", "trawl", "gillnet"]),
            index=0,
        )
        area_type = st.selectbox(
            "Area type",
            options.get("area_type", ["Nearshore Bahrain", "Offshore Bahrain"]),
            index=0,
        )
        bycatch_devices = st.selectbox(
            "Uses bycatch reduction devices?",
            options.get("uses_bycatch_reduction_devices", ["no", "yes"]),
            index=0,
        )
    with c2:
        target_species = st.selectbox(
            "Target species",
            options.get("target_species", ["Crab", "Hamour", "Shrimp"]),
            index=0,
        )
        gear_type = st.selectbox(
            "Gear type",
            options.get("gear_type", ["line", "net", "trap"]),
            index=0,
        )
        enforcement_level = st.selectbox(
            "Enforcement level",
            options.get("enforcement_level", ["low", "medium", "high"]),
            index=0,
        )

    catch_per_trip_kg = st.slider("Catch per trip (kg)", min_value=0, max_value=500, value=120, step=5)

    status_of_target_species = st.selectbox(
        "Status of target species",
        options.get("status_of_target_species", ["healthy", "stable", "declining", "overfished"]),
        index=0,
    )

    st.write("")

    predict = st.button("✅ Check sustainability")

    if predict:
        X_new = pd.DataFrame(
            [
                {
                    "fishing_method": str(fishing_method),
                    "target_species": str(target_species),
                    "area_type": str(area_type),
                    "gear_type": str(gear_type),
                    "uses_bycatch_reduction_devices": str(bycatch_devices),
                    "enforcement_level": str(enforcement_level),
                    "catch_per_trip_kg": float(catch_per_trip_kg),
                    "status_of_target_species": str(status_of_target_species),
                }
            ]
        )

        pred = int(clf.predict(X_new)[0])
        proba = None
        try:
            proba = float(clf.predict_proba(X_new)[0][pred])
        except Exception:
            proba = None

        if pred == 1:
            label = "Likely SUSTAINABLE"
            klass = "good"
            emoji = "🌿"
            desc = "This practice matches patterns that the model learned from examples labeled sustainable."
        else:
            label = "Likely UNSUSTAINABLE"
            klass = "bad"
            emoji = "⚠️"
            desc = "This practice matches patterns that the model learned from examples labeled unsustainable."

        confidence = f" • Confidence: {proba:.0%}" if proba is not None else ""

        st.markdown(
            f"""
<div class="card" style="margin-top: 10px;">
  <div class="pill {klass}">{emoji} {label}{confidence}</div>
  <p style="margin-top: 10px;">{desc}</p>
  <p class="muted">
    Note: This is an educational model trained on example data — not an official scientific or regulatory tool.
  </p>
</div>
""",
            unsafe_allow_html=True,
        )

with right:
    st.markdown(
        """
<div class="card">
  <h2>Step 2 — Understanding the result</h2>
  <ul>
    <li><b>Sustainable</b> means the practice is less likely to harm stock levels or ecosystems in the examples used.</li>
    <li><b>Unsustainable</b> means the pattern resembles examples that risk overfishing, damaging habitats, or weak enforcement.</li>
  </ul>
  <p class="muted">
    Open the pages in the sidebar for: <b>My Story</b>, how the model works, and responsible-use notes.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    acc_text = "—"
    if acc is not None:
        acc_text = f"{acc*100:.1f}%"
    st.markdown(
        f"""
<div class="card">
  <h2>Model snapshot</h2>
  <p><b>Current test accuracy:</b> {acc_text} on held-out data</p>
  <p class="muted">
    Accuracy is a quick check, not a guarantee. Real-world sustainability depends on regulations, seasons, habitats,
    and local ecosystem status.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
