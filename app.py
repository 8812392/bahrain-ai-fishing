import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# ---------------- Theme: Bright Ocean (not gloomy) ----------------
def apply_theme():
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
            --soft:#e8f6ff;
            --danger:#ef4444;
          }

          html, body, [class*="css"]{
            background-color:var(--bg);
            color:var(--text);
          }

          .block-container{
            padding-top:2.0rem;
            padding-bottom:3rem;
            max-width:1150px;
          }

          h1{ font-size:3.0rem !important; margin-bottom:0.2rem; }
          h2{ font-size:1.55rem !important; margin-top:1.2rem; }
          p, li{ font-size:1.06rem !important; line-height:1.75 !important; }

          .hero{
            background:linear-gradient(135deg, rgba(14,165,233,0.13), rgba(20,184,166,0.11));
            border:1px solid var(--border);
            border-radius:22px;
            padding:18px 18px;
            box-shadow:0 10px 28px var(--shadow);
          }

          .subtle{ color:var(--muted); font-size:1.05rem; }

          .card{
            background:var(--card);
            border:1px solid var(--border);
            border-radius:18px;
            padding:18px 18px;
            box-shadow:0 10px 28px var(--shadow);
          }

          .pill{
            display:inline-block;
            padding:6px 10px;
            border-radius:999px;
            border:1px solid var(--border);
            background:#fff;
            margin-right:8px;
            margin-top:8px;
            font-size:0.95rem;
            color:var(--text);
          }

          .resultGood{
            background:rgba(20,184,166,0.12);
            border:1px solid rgba(20,184,166,0.28);
            border-radius:16px;
            padding:14px 14px;
          }

          .resultBad{
            background:rgba(239,68,68,0.10);
            border:1px solid rgba(239,68,68,0.22);
            border-radius:16px;
            padding:14px 14px;
          }

          .tiny{ color:var(--muted); font-size:0.95rem; }
          .divider{ height:1px; background:var(--border); margin:12px 0; }

          .recBox{
            background: rgba(14,165,233,0.08);
            border: 1px solid rgba(14,165,233,0.22);
            border-radius: 16px;
            padding: 14px 14px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------- Helpers (Top-tier upgrades) ----------------
def pretty_label(col: str) -> str:
    """Make dataset column names look human-friendly."""
    s = col.strip().replace("_", " ")
    s = " ".join(s.split())
    # Small title-case improvements
    titled = s.title()
    titled = titled.replace("Kg", "kg").replace("Ai", "AI").replace("Id", "ID")
    titled = titled.replace("Mun", "MUN")
    return titled


def norm_key(s: str) -> str:
    return s.lower().strip().replace(" ", "_")


def get_value(user_input: dict, candidates: list[str]):
    """Find a value in user_input using flexible matching."""
    # Build normalized map
    m = {norm_key(k): user_input.get(k) for k in user_input.keys()}
    for c in candidates:
        ck = norm_key(c)
        # exact match by normalized
        if ck in m:
            return m[ck]
        # fuzzy: contains
        for k in m:
            if ck == k or ck in k or k in ck:
                return m[k]
    return None


def confidence_from_probs(probs: np.ndarray) -> tuple[str, float]:
    """
    Confidence based on probability margin between top two classes.
    Returns (label, margin)
    """
    if probs is None or len(probs) < 2:
        return ("Unknown", 0.0)
    sorted_probs = sorted([float(p) for p in probs], reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]
    # thresholds tuned for “student project” clarity
    if margin >= 0.35:
        return ("High", margin)
    if margin >= 0.18:
        return ("Medium", margin)
    return ("Low", margin)


def build_recommendations(user_input: dict) -> list[str]:
    recs = []

    gear = get_value(user_input, ["gear", "gear_type", "method", "fishing_method"])
    enforcement = get_value(user_input, ["enforcement", "enforcement_level", "law_enforcement"])
    bycatch = get_value(user_input, ["bycatch_reduction", "bycatch", "bycatch_reduction_level"])
    catchkg = get_value(user_input, ["catch_per_trip_kg", "catch per trip kg", "catch_per_trip"])
    depth = get_value(user_input, ["depth", "depth_m", "depth (m)"])
    gear_impact = get_value(user_input, ["gear_impact_score", "gear impact score"])

    gear_s = str(gear).lower() if gear is not None else ""
    enf_s = str(enforcement).lower() if enforcement is not None else ""
    byc_s = str(bycatch).lower() if bycatch is not None else ""

    # 1) Gear/method suggestions
    high_impact_keywords = ["trawl", "trawling", "dredge", "dredging", "drift net", "driftnet", "gillnet"]
    low_impact_alts = "switch to handline, hook-and-line, hand reel, or well-managed traps (where legal)"

    if any(k in gear_s for k in high_impact_keywords):
        recs.append("Consider using a lower-impact fishing method (example: " + low_impact_alts + ").")

    if "illegal" in enf_s:
        recs.append("Your enforcement level is marked as illegal — a key improvement is to follow legal regulations and permitted gear only.")
    elif "low" in enf_s:
        recs.append("If possible, improve compliance: fishing in regulated zones with stronger monitoring often reduces sustainability risk.")

    # 2) Bycatch reduction
    if byc_s in ["none", "low", "very low", "0", "no"]:
        recs.append("Increase bycatch reduction (example: selective hook sizes, circle hooks, escape gaps in traps, avoiding juvenile areas).")

    # 3) Catch size pressure
    if catchkg is not None:
        try:
            ck = float(catchkg)
            if ck >= 700:
                recs.append("Catch per trip is very high — reducing catch size or fishing frequency can lower pressure on fish populations.")
            elif ck >= 450:
                recs.append("Consider moderating catch per trip and prioritizing legal size limits to reduce long-term pressure.")
        except Exception:
            pass

    # 4) Depth / habitat sensitivity
    if depth is not None:
        d = str(depth).lower()
        if any(x in d for x in ["near", "nearshore", "seagrass", "reef"]):
            recs.append("Nearshore habitats can be more sensitive — avoid reefs/seagrass zones and respect protected areas if applicable.")

    # 5) Gear impact score (if you added it)
    if gear_impact is not None:
        try:
            gi = float(gear_impact)
            if gi >= 8:
                recs.append("Your gear impact score is high — choose more selective gear or fish in less sensitive areas to reduce habitat damage.")
        except Exception:
            pass

    # If nothing triggered, still show something useful
    if not recs:
        recs.append("Your scenario already looks relatively responsible — keep focusing on selective gear, reduced bycatch, and respecting sensitive areas.")

    return recs


# ---------------- Data + Model ----------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    possible_labels = ["label", "sustainable", "is_sustainable", "target", "class"]
    label_col = None
    for c in possible_labels:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        label_col = df.columns[-1]

    X = df.drop(columns=[label_col])
    y = df[label_col]

    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=450,
        random_state=42,
        class_weight="balanced",
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    pipe.fit(X_train, y_train)

    acc = 0.0
    if len(X_test) > 0:
        preds = pipe.predict(X_test)
        acc = float(accuracy_score(y_test, preds))

    classes = list(pipe.named_steps["model"].classes_)
    return pipe, acc, label_col, X.columns.tolist(), numeric_features, categorical_features, classes


def safe_default(df: pd.DataFrame, col: str):
    if df[col].dtype.kind in "if":
        return float(df[col].median())
    mode = df[col].mode()
    return str(mode.iloc[0]) if len(mode) else ""


# ---------------- App ----------------
st.set_page_config(page_title="Bahrain Sustainable Fishing AI", page_icon="🌊", layout="wide")
apply_theme()

df = load_data("bahrain_fishing_practices.csv")
pipe, acc, label_col, feature_cols, num_cols, cat_cols, classes = train_model(df)

st.markdown(
    """
    <div class="hero">
      <h1>🌊 Bahrain Sustainable Fishing AI</h1>
      <div class="subtle">
        An educational tool that predicts whether a fishing scenario is more likely to be sustainable — with a score, confidence, and improvement tips.
      </div>
      <div style="margin-top:10px;">
        <span class="pill">🇧🇭 Bahrain</span>
        <span class="pill">🎣 Fishing</span>
        <span class="pill">🤖 AI</span>
        <span class="pill">🪸 Marine protection</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
left, right = st.columns([1.05, 0.95], vertical_alignment="top")


# ---------------- Inputs (prettified labels + no underscores) ----------------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Build a fishing scenario")

    user_input = {}
    for col in feature_cols:
        label = pretty_label(col)
        default = safe_default(df, col)

        if col in num_cols:
            # Your requested range for catch per trip
            if norm_key(col) in ["catch_per_trip_kg", "catch_per_trip_(kg)", "catchpertripkg", "catch_per_trip"]:
                user_input[col] = st.slider(label, min_value=20, max_value=1000,
                                            value=int(default) if default else 200, key=col)
            else:
                mn = float(df[col].min())
                mx = float(df[col].max())
                if np.isfinite(mn) and np.isfinite(mx) and mn != mx:
                    val = float(default) if default is not None else mn
                    user_input[col] = st.slider(label, float(mn), float(mx), float(val), key=col)
                else:
                    user_input[col] = st.number_input(label, value=float(default) if default else 0.0, key=col)

        else:
            options = sorted(df[col].dropna().astype(str).unique().tolist())
            idx = options.index(str(default)) if str(default) in options else 0
            user_input[col] = st.selectbox(label, options, index=idx, key=col)

    run = st.button("Predict sustainability", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- Output (score + confidence + recommendations) ----------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Result")

    st.markdown(
        f"<div class='tiny'>Model test accuracy (quick split): <b>{acc*100:.1f}%</b></div>",
        unsafe_allow_html=True
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if run:
        X_new = pd.DataFrame([user_input], columns=feature_cols)

        probs = None
        proba_sus = None
        if hasattr(pipe, "predict_proba") and len(classes) > 1:
            probs = pipe.predict_proba(X_new)[0]

            # Try to detect which class means “sustainable”
            sus_idx = None
            for i, c in enumerate(classes):
                if str(c).lower() in ["sustainable", "1", "true", "yes"]:
                    sus_idx = i
                    break
            if sus_idx is None:
                sus_idx = 1  # fallback
            proba_sus = float(probs[sus_idx])

        pred = pipe.predict(X_new)[0]
        pred_str = str(pred).lower()
        is_sus = pred_str in ["sustainable", "1", "true", "yes"]

        # Confidence
        conf_label, margin = confidence_from_probs(probs)

        # Score output
        if proba_sus is None:
            if is_sus:
                st.markdown("<div class='resultGood'><h3>✅ Likely Sustainable</h3></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='resultBad'><h3>⚠️ Likely Unsustainable</h3></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='tiny'><b>Confidence:</b> {conf_label}</div>", unsafe_allow_html=True)

        else:
            sus_pct = int(round(proba_sus * 100))
            risk_pct = 100 - sus_pct

            if sus_pct >= 60:
                st.markdown(
                    f"<div class='resultGood'><h3>✅ Sustainability Score: {sus_pct}%</h3>"
                    f"<div class='tiny'>Risk Score: {risk_pct}% • <b>Confidence:</b> {conf_label}</div></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='resultBad'><h3>⚠️ Sustainability Score: {sus_pct}%</h3>"
                    f"<div class='tiny'>Risk Score: {risk_pct}% • <b>Confidence:</b> {conf_label}</div></div>",
                    unsafe_allow_html=True
                )

            st.progress(sus_pct / 100)

            # Confidence explanation (simple and honest)
            st.markdown(
                f"<div class='tiny'>Confidence is based on how clearly the model preferred one class over another (probability gap ≈ {margin:.2f}).</div>",
                unsafe_allow_html=True
            )

        st.write("")
        st.markdown("**How to interpret the result**")
        st.markdown(
            """
            - Higher-impact gear usually increases risk  
            - Sensitive habitats increase risk  
            - Better enforcement and bycatch reduction lowers risk  
            - Very high catch sizes can increase pressure on fish populations  
            """.strip()
        )

        st.write("")
        st.markdown("**Recommendations to improve sustainability**")
        recs = build_recommendations(user_input)
        st.markdown("<div class='recBox'>", unsafe_allow_html=True)
        for r in recs:
            st.markdown(f"- {r}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown(
            "<div class='tiny'>Educational note: this is a student model trained on sample data. It helps learning and discussion, not official decisions.</div>",
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            "<div class='tiny'>Fill the scenario on the left and click <b>Predict sustainability</b>.</div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.markdown(
    """
    <div class="card">
      <b>Tip:</b> Use the sidebar to open <b>My Story</b> for the personal meaning behind this project.
    </div>
    """,
    unsafe_allow_html=True
)
