import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_PATH    = "best_model.pkl"
SCALER_PATH   = "scaler.pkl"
SELECTOR_PATH = "selector.pkl"
DATA_PATH     = "HousePricePrediction.csv"
K_BEST        = 5

INPUT_FEATURES = [
    "MSSubClass", "LotArea", "OverallCond",
    "YearBuilt", "YearRemodAdd", "BsmtFinSF2", "TotalBsmtSF"
]

# ─────────────────────────────────────────────
# ML Pipeline
# ─────────────────────────────────────────────
def train_pipeline():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset '{DATA_PATH}' not found. "
            "Place HousePricePrediction.csv in the same directory as this script."
        )

    df = pd.read_csv(DATA_PATH)

    # Impute
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    num_imputer = SimpleImputer(strategy="mean")
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Outlier removal
    num_df = df.select_dtypes(include=["int64", "float64"])
    Q1, Q3 = num_df.quantile(0.25), num_df.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)
    df = df.loc[mask]

    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)

    # Split X / y
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    # Feature selection
    k = min(K_BEST, X.shape[1])
    selector = SelectKBest(score_func=f_regression, k=k)
    X_sel = selector.fit_transform(X, y)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Train models
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(),
        "Random Forest":     RandomForestRegressor(),
    }

    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores[name] = r2_score(y_test, model.predict(X_test))

    best_name  = max(scores, key=scores.get)
    best_model = models[best_name]

    pickle.dump(best_model, open(MODEL_PATH,    "wb"))
    pickle.dump(scaler,     open(SCALER_PATH,   "wb"))
    pickle.dump(selector,   open(SELECTOR_PATH, "wb"))

    return scores, best_name


def load_artifacts():
    model    = pickle.load(open(MODEL_PATH,    "rb"))
    scaler   = pickle.load(open(SCALER_PATH,   "rb"))
    selector = pickle.load(open(SELECTOR_PATH, "rb"))
    return model, scaler, selector


def predict_price(features: dict):
    model, scaler, selector = load_artifacts()

    df = pd.read_csv(DATA_PATH)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    num_imputer = SimpleImputer(strategy="mean")
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    num_df = df.select_dtypes(include=["int64", "float64"])
    Q1, Q3 = num_df.quantile(0.25), num_df.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)
    df = df.loc[mask]

    df = pd.get_dummies(df, drop_first=True)
    target = df.columns[-1]
    X_all = df.drop(columns=[target])

    row = pd.DataFrame([{col: 0 for col in X_all.columns}])
    for col, val in features.items():
        if col in row.columns:
            row[col] = val

    X_sel = selector.transform(row)
    X_sc  = scaler.transform(X_sel)
    pred  = model.predict(X_sc)[0]
    return max(pred, 0)


def models_exist():
    return (
        os.path.exists(MODEL_PATH) and
        os.path.exists(SCALER_PATH) and
        os.path.exists(SELECTOR_PATH)
    )


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.markdown("""
    <style>
        .main { max-width: 640px; }
        .stNumberInput > div > div > input { font-size: 15px; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("##### ML-Powered Valuation")
st.title("🏠 House Price Predictor")
st.caption(
    "Enter property details below. The model is trained on the Ames Housing dataset "
    "using the best-performing regression algorithm."
)

st.divider()

# ── Input Form ────────────────────────────────
st.subheader("Property Details")

col1, col2 = st.columns(2)

with col1:
    ms_subclass = st.selectbox(
        "MS SubClass",
        options=[20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190],
        format_func=lambda x: {
            20:  "20 — 1-Story 1946+",
            30:  "30 — 1-Story 1945 & Older",
            40:  "40 — 1-Story w/ Finished Attic",
            45:  "45 — 1.5-Story Unfinished",
            50:  "50 — 1.5-Story Finished",
            60:  "60 — 2-Story 1946+",
            70:  "70 — 2-Story 1945 & Older",
            75:  "75 — 2.5-Story All Ages",
            80:  "80 — Split or Multi-Level",
            85:  "85 — Split Foyer",
            90:  "90 — Duplex",
            120: "120 — 1-Story PUD 1946+",
            150: "150 — 1.5-Story PUD",
            160: "160 — 2-Story PUD 1946+",
            180: "180 — PUD Multilevel",
            190: "190 — 2 Family Conversion",
        }.get(x, str(x)),
        index=0
    )

    overall_cond = st.number_input(
        "Overall Condition (1–10)",
        min_value=1, max_value=10, value=5, step=1
    )

    year_remod = st.number_input(
        "Year Remodelled",
        min_value=1950, max_value=2025, value=2000, step=1
    )

    total_bsmt_sf = st.number_input(
        "Total Basement SF",
        min_value=0, max_value=6000, value=800, step=10
    )

with col2:
    lot_area = st.number_input(
        "Lot Area (sq ft)",
        min_value=500, max_value=200000, value=8000, step=100
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1800, max_value=2025, value=1990, step=1
    )

    bsmt_fin_sf2 = st.number_input(
        "Basement Finished SF Type 2",
        min_value=0, max_value=2000, value=0, step=10
    )

st.divider()

# ── Predict Button ────────────────────────────
if st.button("🔍 Estimate Price", use_container_width=True, type="primary"):
    if not models_exist():
        st.error("⚠️ Model not trained yet. Use the **Retrain Model** section below to train first.")
    elif not os.path.exists(DATA_PATH):
        st.error(f"⚠️ Dataset `{DATA_PATH}` not found. Place it in the same folder as this script.")
    else:
        with st.spinner("Estimating price…"):
            try:
                features = {
                    "MSSubClass":   float(ms_subclass),
                    "LotArea":      float(lot_area),
                    "OverallCond":  float(overall_cond),
                    "YearBuilt":    float(year_built),
                    "YearRemodAdd": float(year_remod),
                    "BsmtFinSF2":   float(bsmt_fin_sf2),
                    "TotalBsmtSF":  float(total_bsmt_sf),
                }
                price = predict_price(features)
                model_name = type(pickle.load(open(MODEL_PATH, "rb"))).__name__

                st.success("Prediction complete!")
                st.metric(
                    label="Estimated Sale Price",
                    value=f"${price:,.0f}"
                )
                st.caption(f"Model used: `{model_name}`")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.divider()

# ── Retrain Section ───────────────────────────
with st.expander("↺ Retrain Model"):
    st.caption("Retrain all three models on the dataset and save the best one.")

    if st.button("Start Training", use_container_width=True):
        if not os.path.exists(DATA_PATH):
            st.error(f"Dataset `{DATA_PATH}` not found. Place it in the same folder as this script.")
        else:
            with st.spinner("Training models…"):
                try:
                    scores, best_name = train_pipeline()
                    st.success(f"✅ Training complete! Best model: **{best_name}**")

                    results_df = pd.DataFrame(
                        {"Model": list(scores.keys()), "R² Score": [round(v, 4) for v in scores.values()]}
                    ).sort_values("R² Score", ascending=False).reset_index(drop=True)

                    st.dataframe(results_df, use_container_width=True)

                except Exception as e:
                    st.error(f"Training failed: {e}")
