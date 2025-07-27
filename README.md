
# 🏗️ Bulldozer Sale Price Prediction using Machine Learning

This project builds a regression model to predict the sale prices of heavy machinery (specifically bulldozers) using historical auction data, enhanced with temporal features and categorical encoding strategies.

---

## 📌 Project Overview

- **Objective**: Predict bulldozer sale prices using prior sales data.  
- **Problem Type**: Supervised Regression  
- **Domain**: Heavy Equipment / Auction Pricing Analytics  
- **Target Variable**: `SalePrice`  
- **Tech Stack**: Python, Jupyter Notebook, pandas, NumPy, matplotlib, seaborn, scikit-learn

---

## 📂 Dataset Description

- **TrainAndValid.csv** – ~412k rows, 53 columns including:
  - Identifier data (`SalesID`, `MachineID`, `ModelID`, `datasource`, `auctioneerID`)
  - Equipment features (`YearMade`, `UsageBand`, `ProductSize`, etc.)
  - `saledate` parsed as `datetime64`
  - Target field: `SalePrice`
- Includes a separate **Test.csv** for final predictions.

---

## 🔄 Workflow

### 1. **Initial Data Exploration**
- Inspect data types and summary statistics
- Visualize sale price trends over time and by month
- Explore geographic variation using `state`

### 2. **Time-Series Feature Engineering**
- Convert `saledate` into:  
  `saleYear`, `saleMonth`, `saleDay`, `saleDayofweek`, `saleDayofyear`
- Remove the original `saledate` column

### 3. **Handling Missing Values & Encoding**
- **Numericals**: impute missing values with median + add `_is_missing` flag  
- **Categoricals**: convert to `category` type, and encode as integers (codes +1), retaining missing flags  
- Handle inconsistencies between train and test sets (e.g. ensuring `auctioneerID_is_missing` exists)

### 4. **Train-Validation Split**
- Investments from year 2012 as validation (~11k rows), and earlier years as training (~401k rows)

### 5. **Model Training & Evaluation**
- **Base Model**: RandomForestRegressor with `max_samples=10000`  
   → Initial results:  
   - Training MAE ≈ 5,550  
   - Valid MAE ≈ 7,097  
   - Valid RMSLE ≈ 0.2906  
   - Valid R² ≈ 0.836  

- **Hyperparameter Tuning** using `RandomizedSearchCV` (20 candidates, 5‑fold CV)  
   → Best tuning gave slightly worse validation performance (MAE ≈ 7,568, RMSLE ≈ 0.305)

- **Final “Ideal” Model** with selected hyperparameters:
  ```
  n_estimators=90, max_features=0.5, min_samples_split=14, min_samples_leaf=1, max_samples=None
  ```
  - Training MAE ≈ 2,930  
  - Valid MAE ≈ 5,911  
  - Valid RMSLE ≈ 0.2438  
  - Valid R² ≈ 0.884  

### 6. **Prediction on Test Set**
- Preprocess test data using same logic (adding missing flags, encoding categories)
- Align feature columns with training set
- Generate `SalePrice` predictions for submission with `SalesID`

### 7. **Feature Importance Interpretation**
- Displayed top 20 model features using a barplot  
- Confirmed that importance scores sum to 1.0

---

## 📈 Results Summary

| Model                 | Training MAE | Validation MAE | Valid RMSLE | Valid R²  |
|-----------------------|--------------|----------------|-------------|-----------|
| Base Random Forest    | ~5,550        | ~7,097          | ~0.291      | ~0.836    |
| Tuned (RS‑CV)         | ~6,050        | ~7,568          | ~0.305      | ~0.807    |
| Final “Ideal” Model   | ~2,930        | ~5,911          | ~0.244      | ~0.884    |

---

## 📎 Requirements

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 🚀 How to Run

1. Clone or download the notebook and data (`TrainAndValid.csv`, `Test.csv`)
2. Install dependencies
3. Open and run the notebook in order:
   - Data loading & exploratory analysis
   - Feature engineering & preprocessing
   - Model training, evaluation & tuning
   - Generating predictions for test data

---

## 📁 Suggested Project Structure

```
Bulldozer_Price_Prediction/
│
├── TrainAndValid.csv
├── Test.csv
├── Bulldozer_SalePrice_Prediction.ipynb  # Main notebook
├── model/                                # (Optional) Saved trained model
└── README.md
```

---

## ✅ Future Enhancements

- Optimize preprocessing (e.g. better imputation, outlier handling)
- Add feature selection or dimensionality reduction
- Try ensemble models (e.g., XGBoost, LightGBM) or stacking
- Use more sophisticated hyperparameter search (e.g. Bayesian optimization)
- Deploy interactive app via Streamlit or Flask
- Explore time-series cross‑validation for temporally ordered data

---

## 👨‍💻 Author

**Deepak P.**  
Machine Learning Engineer | Predictive Analytics Enthusiast  
📧 deepak@email.com | 🔗 LinkedIn | 🌐 Portfolio (optional)
