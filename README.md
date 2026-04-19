# 🏠 House Price Prediction — Streamlit App

A machine-learning web app that predicts house sale prices using the Ames Housing dataset.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## 📋 Steps

1. **Train Model** → Go to the `🤖 Train Model` tab and click **Start Training**
2. **Predict Price** → Fill in property details in `🔍 Predict Price` and click Estimate
3. **Explore Data** → View charts and correlations in `📊 EDA`

---

## 📁 File Structure

```
HousePricePrediction/
├── app.py                    ← Streamlit app (all pages)
├── HousePricePrediction.csv  ← Dataset (2919 rows, 13 columns)
├── requirements.txt          ← Python dependencies
├── best_model.pkl            ← Auto-generated after training
├── scaler.pkl                ← Auto-generated after training
└── train_columns.pkl         ← Auto-generated after training
```

---

## 🤖 Models

| Model | Type |
|---|---|
| Linear Regression | Baseline |
| Decision Tree | Non-linear |
| Random Forest | Ensemble (usually best) |

The best model (by R²) is saved automatically.

---

## 📊 Features Used for Prediction

- MSSubClass, LotArea, OverallCond
- YearBuilt, YearRemodAdd
- BsmtFinSF2, TotalBsmtSF
