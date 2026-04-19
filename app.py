import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────────────────────
# Paths & Constants
# ─────────────────────────────────────────────────────────────
MODEL_PATH   = "best_model.pkl"
SCALER_PATH  = "scaler.pkl"
COLS_PATH    = "train_columns.pkl"
DATA_PATH    = "HousePricePrediction.csv"

FEATURE_COLS = [
    "MSSubClass", "LotArea", "OverallCond",
    "YearBuilt", "YearRemodAdd", "BsmtFinSF2", "TotalBsmtSF"
]

MS_SUBCLASS_MAP = {
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
}

# ─────────────────────────────────────────────────────────────
# ML Pipeline
# ─────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["SalePrice"])
    return df


def preprocess(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    num_imp = SimpleImputer(strategy="mean")
    df[num_cols] = num_imp.fit_transform(df[num_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    df = pd.get_dummies(df, drop_first=True)
    return df


def train_pipeline():
    df_raw = load_data()
    df = preprocess(df_raw)

    if "SalePrice" not in df.columns:
        raise ValueError("SalePrice column not found after preprocessing.")

    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice", "Id"], errors="ignore")

    # Keep only the 7 selected features that are present
    available = [c for c in FEATURE_COLS if c in X.columns]
    X = X[available]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree":     DecisionTreeRegressor(random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        preds = model.predict(X_test_sc)
        results[name] = {
            "model":  model,
            "r2":     r2_score(y_test, preds),
            "mae":    mean_absolute_error(y_test, preds),
            "rmse":   np.sqrt(mean_squared_error(y_test, preds)),
            "preds":  preds,
            "y_test": y_test.values,
        }

    best_name = max(results, key=lambda k: results[k]["r2"])
    best_model = results[best_name]["model"]

    pickle.dump(best_model,   open(MODEL_PATH,  "wb"))
    pickle.dump(scaler,        open(SCALER_PATH, "wb"))
    pickle.dump(available,     open(COLS_PATH,   "wb"))

    return results, best_name


def artifacts_exist():
    return (
        os.path.exists(MODEL_PATH) and
        os.path.exists(SCALER_PATH) and
        os.path.exists(COLS_PATH)
    )


def predict_price(input_dict: dict) -> float:
    model   = pickle.load(open(MODEL_PATH,  "rb"))
    scaler  = pickle.load(open(SCALER_PATH, "rb"))
    columns = pickle.load(open(COLS_PATH,   "rb"))

    row = pd.DataFrame([{c: input_dict.get(c, 0) for c in columns}])
    row_sc = scaler.transform(row)
    pred   = model.predict(row_sc)[0]
    return max(pred, 0)


# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

st.markdown("""
<style>
    .metric-box {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 14px 20px;
        text-align: center;
    }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Sidebar – Navigation
# ─────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/home.png", width=60)
st.sidebar.title("🏠 House Price App")
page = st.sidebar.radio(
    "Navigate",
    ["🔍 Predict Price", "📊 EDA", "🤖 Train Model", "📄 About"],
    index=0,
)
st.sidebar.divider()
st.sidebar.caption("Powered by scikit-learn + Streamlit")

# ─────────────────────────────────────────────────────────────
# Page 1 – Predict
# ─────────────────────────────────────────────────────────────
if page == "🔍 Predict Price":
    st.title("🏠 House Price Predictor")
    st.caption("Fill in the property details and get an instant ML-powered price estimate.")

    if not artifacts_exist():
        st.warning("⚠️ Model not trained yet. Go to **🤖 Train Model** tab first.")
        st.stop()

    st.subheader("Property Details")

    col1, col2 = st.columns(2)

    with col1:
        ms_subclass = st.selectbox(
            "MS SubClass (Building Type)",
            options=list(MS_SUBCLASS_MAP.keys()),
            format_func=lambda x: MS_SUBCLASS_MAP[x],
        )
        overall_cond = st.slider("Overall Condition (1–10)", 1, 10, 5)
        year_remod   = st.number_input("Year Remodelled", 1950, 2025, 2000, step=1)
        total_bsmt   = st.number_input("Total Basement SF", 0, 6000, 800, step=10)

    with col2:
        lot_area     = st.number_input("Lot Area (sq ft)", 500, 200_000, 8_000, step=100)
        year_built   = st.number_input("Year Built", 1800, 2025, 1990, step=1)
        bsmt_fin_sf2 = st.number_input("Basement Finished SF Type 2", 0, 2000, 0, step=10)

    st.divider()

    if st.button("🔍 Estimate Price", use_container_width=True, type="primary"):
        with st.spinner("Computing estimate…"):
            try:
                inp = {
                    "MSSubClass":   float(ms_subclass),
                    "LotArea":      float(lot_area),
                    "OverallCond":  float(overall_cond),
                    "YearBuilt":    float(year_built),
                    "YearRemodAdd": float(year_remod),
                    "BsmtFinSF2":   float(bsmt_fin_sf2),
                    "TotalBsmtSF":  float(total_bsmt),
                }
                price = predict_price(inp)
                model_name = type(pickle.load(open(MODEL_PATH, "rb"))).__name__

                st.success("✅ Prediction Complete!")
                c1, c2, c3 = st.columns(3)
                c1.metric("💰 Estimated Sale Price", f"${price:,.0f}")
                c2.metric("🤖 Model Used", model_name.replace("Regressor", ""))
                c3.metric("📐 Lot Area", f"{lot_area:,} sq ft")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ─────────────────────────────────────────────────────────────
# Page 2 – EDA
# ─────────────────────────────────────────────────────────────
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    if not os.path.exists(DATA_PATH):
        st.error(f"`{DATA_PATH}` not found.")
        st.stop()

    df = load_data()
    st.subheader("Dataset Overview")

    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Rows", f"{df.shape[0]:,}")
    r1c2.metric("Columns", df.shape[1])
    r1c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    # Sale Price distribution
    st.subheader("Sale Price Distribution")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.histplot(df["SalePrice"].dropna(), bins=40, kde=True, color="#4f8ef7", ax=ax)
    ax.set_xlabel("Sale Price ($)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Sale Prices")
    st.pyplot(fig)

    st.divider()

    # Correlation
    st.subheader("Feature Correlation with Sale Price")
    num_df = df.select_dtypes(include=["int64", "float64"])
    corr = num_df.corr()["SalePrice"].drop("SalePrice").sort_values(ascending=False)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    corr.plot(kind="bar", color=["#4f8ef7" if v >= 0 else "#f74f4f" for v in corr], ax=ax2)
    ax2.set_title("Correlation with SalePrice")
    ax2.set_ylabel("Pearson r")
    ax2.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig2)

    st.divider()

    # Scatter: YearBuilt vs SalePrice
    st.subheader("Year Built vs Sale Price")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.scatter(df["YearBuilt"], df["SalePrice"], alpha=0.4, s=15, color="#4f8ef7")
    ax3.set_xlabel("Year Built")
    ax3.set_ylabel("Sale Price ($)")
    ax3.set_title("Year Built vs Sale Price")
    st.pyplot(fig3)

    # Scatter: LotArea vs SalePrice
    st.subheader("Lot Area vs Sale Price")
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.scatter(df["LotArea"], df["SalePrice"], alpha=0.4, s=15, color="#f7a24f")
    ax4.set_xlabel("Lot Area (sq ft)")
    ax4.set_ylabel("Sale Price ($)")
    ax4.set_title("Lot Area vs Sale Price")
    st.pyplot(fig4)

# ─────────────────────────────────────────────────────────────
# Page 3 – Train Model
# ─────────────────────────────────────────────────────────────
elif page == "🤖 Train Model":
    st.title("🤖 Train the Model")
    st.caption("Trains three regression models and saves the best-performing one.")

    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset `{DATA_PATH}` not found. Place it in the same folder as `app.py`.")
        st.stop()

    st.info("Click **Start Training** to fit Linear Regression, Decision Tree, and Random Forest on the dataset.")

    if st.button("▶ Start Training", type="primary", use_container_width=True):
        with st.spinner("Training models… this may take a moment."):
            try:
                results, best_name = train_pipeline()
                st.success(f"✅ Training complete! Best model: **{best_name}**")
                st.divider()

                # Metrics table
                st.subheader("Model Comparison")
                rows = []
                for name, res in results.items():
                    rows.append({
                        "Model":     name,
                        "R² Score":  round(res["r2"], 4),
                        "MAE ($)":   f"{res['mae']:,.0f}",
                        "RMSE ($)":  f"{res['rmse']:,.0f}",
                        "Best":      "⭐" if name == best_name else "",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                # Actual vs Predicted plot for best model
                st.subheader(f"Actual vs Predicted — {best_name}")
                best = results[best_name]
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(best["y_test"], best["preds"], alpha=0.5, s=18, color="#4f8ef7")
                mn = min(best["y_test"].min(), best["preds"].min())
                mx = max(best["y_test"].max(), best["preds"].max())
                ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
                ax.set_xlabel("Actual Price ($)")
                ax.set_ylabel("Predicted Price ($)")
                ax.set_title(f"{best_name} — Actual vs Predicted")
                ax.legend()
                st.pyplot(fig)

                # Feature importance (Random Forest only)
                if best_name == "Random Forest":
                    st.subheader("Feature Importances")
                    cols = pickle.load(open(COLS_PATH, "rb"))
                    importances = best["model"].feature_importances_
                    imp_df = pd.DataFrame({"Feature": cols, "Importance": importances})
                    imp_df = imp_df.sort_values("Importance", ascending=True)
                    fig2, ax2 = plt.subplots(figsize=(7, 4))
                    ax2.barh(imp_df["Feature"], imp_df["Importance"], color="#4f8ef7")
                    ax2.set_title("Feature Importances (Random Forest)")
                    st.pyplot(fig2)

            except Exception as e:
                st.error(f"Training failed: {e}")

# ─────────────────────────────────────────────────────────────
# Page 4 – About
# ─────────────────────────────────────────────────────────────
elif page == "📄 About":
    st.title("📄 About This Project")

    st.markdown("""
    ## 🏠 House Price Prediction

    This Streamlit app predicts residential house sale prices using machine learning
    regression models trained on the **Ames Housing dataset**.

    ---

    ### 🔧 Features Used
    | Feature | Description |
    |---|---|
    | MSSubClass | Type of dwelling |
    | LotArea | Lot size in square feet |
    | OverallCond | Overall material and finish condition (1–10) |
    | YearBuilt | Year the house was originally constructed |
    | YearRemodAdd | Year of last remodel or addition |
    | BsmtFinSF2 | Finished basement area (type 2) |
    | TotalBsmtSF | Total basement square footage |

    ---

    ### 🤖 Models Trained
    - **Linear Regression** — Fast baseline, assumes linear relationships
    - **Decision Tree** — Non-linear, interpretable splits
    - **Random Forest** — Ensemble of trees; typically best accuracy

    ---

    ### 🚀 How to Run

    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```

    1. Go to **🤖 Train Model** to train and save the best model
    2. Go to **🔍 Predict Price** to get a house price estimate
    3. Explore **📊 EDA** for visual insights

    ---

    ### 📁 Project Structure
    ```
    HousePricePrediction/
    ├── app.py                    ← Main Streamlit app
    ├── HousePricePrediction.csv  ← Dataset
    ├── requirements.txt          ← Python dependencies
    ├── best_model.pkl            ← Saved after training
    ├── scaler.pkl                ← Saved after training
    └── train_columns.pkl         ← Saved after training
    ```
    """)
