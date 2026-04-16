import pandas as pd
import numpy as np
import pickle
import os
from flask import Flask, request, jsonify, render_template_string

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

app = Flask(__name__)

# ─────────────────────────────────────────────
# HTML Template
# ─────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>House Price Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0d0d0d;
    --surface: #161616;
    --border: #2a2a2a;
    --accent: #c8a96e;
    --accent2: #7ecac3;
    --text: #e8e0d4;
    --muted: #6b6560;
    --danger: #e06c6c;
    --success: #6ec8a9;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Mono', monospace;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px 24px 80px;
    background-image:
      radial-gradient(ellipse 60% 40% at 70% 10%, rgba(200,169,110,0.06) 0%, transparent 60%),
      radial-gradient(ellipse 50% 35% at 20% 80%, rgba(126,202,195,0.05) 0%, transparent 55%);
  }

  header {
    text-align: center;
    margin-bottom: 48px;
    animation: fadeDown 0.6s ease both;
  }

  header .eyebrow {
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 12px;
  }

  header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.4rem, 6vw, 4rem);
    font-style: italic;
    line-height: 1.1;
    color: var(--text);
  }

  header p {
    margin-top: 14px;
    color: var(--muted);
    font-size: 13px;
    max-width: 420px;
    margin-inline: auto;
    line-height: 1.7;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 36px 40px;
    width: 100%;
    max-width: 560px;
    animation: fadeUp 0.6s ease 0.15s both;
  }

  .section-label {
    font-size: 10px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }

  .field { margin-bottom: 22px; }

  label {
    display: block;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 8px;
  }

  input, select {
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 14px;
    padding: 12px 14px;
    outline: none;
    transition: border-color 0.2s;
  }

  input:focus, select:focus { border-color: var(--accent); }
  select option { background: #1e1e1e; }

  .btn {
    width: 100%;
    margin-top: 8px;
    padding: 15px;
    background: var(--accent);
    color: #0d0d0d;
    border: none;
    border-radius: 10px;
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s;
  }

  .btn:hover { opacity: 0.88; transform: translateY(-1px); }
  .btn:active { transform: translateY(0); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

  .result-box {
    margin-top: 28px;
    padding: 24px;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.025);
    display: none;
    animation: fadeUp 0.4s ease both;
  }

  .result-box.visible { display: block; }
  .result-box.error { border-color: var(--danger); }
  .result-box.success { border-color: var(--success); }

  .result-label {
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
  }

  .result-price {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: var(--accent);
    line-height: 1;
  }

  .result-meta {
    margin-top: 12px;
    font-size: 12px;
    color: var(--muted);
  }

  .result-error {
    color: var(--danger);
    font-size: 13px;
    line-height: 1.6;
  }

  .train-btn-wrap {
    margin-top: 28px;
    padding-top: 24px;
    border-top: 1px solid var(--border);
  }

  .train-btn {
    width: 100%;
    padding: 13px;
    background: transparent;
    color: var(--accent2);
    border: 1px solid var(--accent2);
    border-radius: 10px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    cursor: pointer;
    transition: background 0.2s, color 0.2s;
  }

  .train-btn:hover { background: rgba(126,202,195,0.08); }

  .train-log {
    margin-top: 18px;
    padding: 16px;
    border-radius: 10px;
    background: rgba(0,0,0,0.3);
    border: 1px solid var(--border);
    font-size: 12px;
    line-height: 1.9;
    display: none;
    white-space: pre-wrap;
    color: var(--text);
  }

  .train-log.visible { display: block; }

  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: rgba(200,169,110,0.12);
    color: var(--accent);
    margin-left: 8px;
    vertical-align: middle;
  }

  @keyframes fadeDown {
    from { opacity: 0; transform: translateY(-18px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
  }
</style>
</head>
<body>

<header>
  <div class="eyebrow">ML-Powered Valuation</div>
  <h1>House Price<br/>Predictor</h1>
  <p>Enter property details below. The model is trained on the Ames Housing dataset using the best-performing regression algorithm.</p>
</header>

<div class="card">
  <div class="section-label">Property Details</div>

  <div class="field">
    <label>MS SubClass</label>
    <select id="MSSubClass">
      <option value="20">20 — 1-Story 1946+</option>
      <option value="30">30 — 1-Story 1945 &amp; Older</option>
      <option value="40">40 — 1-Story w/ Finished Attic</option>
      <option value="45">45 — 1.5-Story Unfinished</option>
      <option value="50">50 — 1.5-Story Finished</option>
      <option value="60" selected>60 — 2-Story 1946+</option>
      <option value="70">70 — 2-Story 1945 &amp; Older</option>
      <option value="75">75 — 2.5-Story All Ages</option>
      <option value="80">80 — Split or Multi-Level</option>
      <option value="90">90 — Duplex All Styles</option>
      <option value="120">120 — PUD 1-Story 1946+</option>
      <option value="160">160 — PUD 2-Story 1946+</option>
      <option value="190">190 — 2-Family Conversion</option>
    </select>
  </div>

  <div class="field">
    <label>Lot Area (sq ft)</label>
    <input type="number" id="LotArea" value="8500" min="1000" max="220000"/>
  </div>

  <div class="field">
    <label>Overall Condition <span class="badge">1–9</span></label>
    <input type="number" id="OverallCond" value="5" min="1" max="9"/>
  </div>

  <div class="field">
    <label>Year Built</label>
    <input type="number" id="YearBuilt" value="2000" min="1872" max="2010"/>
  </div>

  <div class="field">
    <label>Year Remodeled / Added</label>
    <input type="number" id="YearRemodAdd" value="2000" min="1950" max="2010"/>
  </div>

  <div class="field">
    <label>Basement Finished SF 2</label>
    <input type="number" id="BsmtFinSF2" value="0" min="0"/>
  </div>

  <div class="field">
    <label>Total Basement SF</label>
    <input type="number" id="TotalBsmtSF" value="1000" min="0"/>
  </div>

  <button class="btn" id="predictBtn" onclick="predict()">Estimate Price</button>

  <div class="result-box" id="resultBox">
    <div class="result-label">Estimated Sale Price</div>
    <div class="result-price" id="resultPrice"></div>
    <div class="result-meta" id="resultMeta"></div>
    <div class="result-error" id="resultError"></div>
  </div>

  <div class="train-btn-wrap">
    <button class="train-btn" onclick="trainModel()">↺ Retrain Model</button>
    <div class="train-log" id="trainLog"></div>
  </div>
</div>

<script>
async function predict() {
  const btn = document.getElementById('predictBtn');
  btn.disabled = true;
  btn.textContent = 'Estimating…';

  const payload = {
    MSSubClass:   parseFloat(document.getElementById('MSSubClass').value),
    LotArea:      parseFloat(document.getElementById('LotArea').value),
    OverallCond:  parseFloat(document.getElementById('OverallCond').value),
    YearBuilt:    parseFloat(document.getElementById('YearBuilt').value),
    YearRemodAdd: parseFloat(document.getElementById('YearRemodAdd').value),
    BsmtFinSF2:   parseFloat(document.getElementById('BsmtFinSF2').value),
    TotalBsmtSF:  parseFloat(document.getElementById('TotalBsmtSF').value),
  };

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    showResult(data);
  } catch (e) {
    showResult({ error: 'Network error: ' + e.message });
  } finally {
    btn.disabled = false;
    btn.textContent = 'Estimate Price';
  }
}

async function trainModel() {
  const log = document.getElementById('trainLog');
  log.textContent = 'Training model…';
  log.classList.add('visible');

  try {
    const res = await fetch('/train', { method: 'POST' });
    const data = await res.json();
    if (data.error) {
      log.textContent = '✗ Error: ' + data.error;
    } else {
      let out = '✓ Training complete\n\n';
      for (const [m, s] of Object.entries(data.scores)) {
        out += `${m.padEnd(24)} R² = ${s.toFixed(4)}\n`;
      }
      out += `\n★ Best model: ${data.best_model}`;
      log.textContent = out;
    }
  } catch (e) {
    log.textContent = '✗ Network error: ' + e.message;
  }
}

function showResult(data) {
  const box = document.getElementById('resultBox');
  const price = document.getElementById('resultPrice');
  const meta = document.getElementById('resultMeta');
  const err = document.getElementById('resultError');

  box.classList.remove('success', 'error');
  price.textContent = '';
  meta.textContent = '';
  err.textContent = '';

  if (data.error) {
    box.classList.add('error', 'visible');
    err.textContent = data.error;
  } else {
    box.classList.add('success', 'visible');
    price.textContent = '$' + Number(data.predicted_price).toLocaleString('en-US', { maximumFractionDigits: 0 });
    meta.textContent = `Model: ${data.model}`;
  }
}
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────
# ML Pipeline
# ─────────────────────────────────────────────
MODEL_PATH  = "best_model.pkl"
SCALER_PATH = "scaler.pkl"
SELECTOR_PATH = "selector.pkl"
DATA_PATH   = "HousePricePrediction.csv"

NUMERIC_FEATURES = [
    "Id", "MSSubClass", "LotArea", "OverallCond",
    "YearBuilt", "YearRemodAdd", "BsmtFinSF2", "TotalBsmtSF", "SalePrice"
]

INPUT_FEATURES = [
    "MSSubClass", "LotArea", "OverallCond",
    "YearBuilt", "YearRemodAdd", "BsmtFinSF2", "TotalBsmtSF"
]

K_BEST = 5  # SelectKBest k


def train_pipeline():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset '{DATA_PATH}' not found. "
            "Place HousePricePrediction.csv in the same directory as app.py."
        )

    df = pd.read_csv(DATA_PATH)

    # ── Impute ──────────────────────────────
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    num_imputer = SimpleImputer(strategy="mean")
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # ── Outlier removal (numeric only) ──────
    num_df = df.select_dtypes(include=["int64", "float64"])
    Q1, Q3 = num_df.quantile(0.25), num_df.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).any(axis=1)
    df = df.loc[mask]

    # ── One-hot encode ───────────────────────
    df = pd.get_dummies(df, drop_first=True)

    # ── Split X / y ─────────────────────────
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    # ── Feature selection ───────────────────
    k = min(K_BEST, X.shape[1])
    selector = SelectKBest(score_func=f_regression, k=k)
    X_sel = selector.fit_transform(X, y)

    # ── Train / test split ──────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=0.2, random_state=42
    )

    # ── Scale ───────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Train models ────────────────────────
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

    # ── Save artifacts ──────────────────────
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

    # Build a full-feature row matching what was used during training
    # We must reconstruct a DataFrame with the same columns as training X
    # Since we only have raw numeric inputs here, we replicate the pipeline
    # on a single-row dummy dataset.

    # Re-load and preprocess the training data to get column structure
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

    # Build a one-row input aligned to X_all columns
    row = pd.DataFrame([{col: 0 for col in X_all.columns}])
    for col, val in features.items():
        if col in row.columns:
            row[col] = val

    X_sel = selector.transform(row)
    X_sc  = scaler.transform(X_sel)
    pred  = model.predict(X_sc)[0]
    return max(pred, 0)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/train", methods=["POST"])
def train():
    try:
        scores, best = train_pipeline()
        return jsonify({"scores": scores, "best_model": best})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(SELECTOR_PATH)):
            return jsonify({"error": "Model not trained yet. Click 'Retrain Model' first."}), 400

        features = request.get_json(force=True)
        price = predict_price(features)
        model_name = type(pickle.load(open(MODEL_PATH, "rb"))).__name__
        return jsonify({"predicted_price": round(float(price), 2), "model": model_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  House Price Prediction App")
    print("=" * 55)
    print("  Place HousePricePrediction.csv in the same folder.")
    print("  Then visit: http://127.0.0.1:5000")
    print("  Click 'Retrain Model' the first time to train.\n")
    app.run(debug=True, port=5000)
