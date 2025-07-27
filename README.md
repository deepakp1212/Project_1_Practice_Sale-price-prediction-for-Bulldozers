
# 🏷️ Sales Price Prediction using Regression

This project focuses on building a machine learning regression model to predict the **sale price of products** based on various features such as item weight, item type, outlet information, and more. This is a practical, real-world example of applying regression algorithms in retail data analytics.

---

## 📌 Project Overview

- **Objective**: Predict the sales price of individual items in a retail dataset.
- **Problem Type**: Supervised Machine Learning – Regression
- **Domain**: Retail, Pricing Analytics
- **Target Variable**: `Item_Outlet_Sales`
- **Tech Stack**: Python, Jupyter Notebook, scikit-learn, pandas, seaborn, matplotlib

---

## 📂 Dataset Description

The dataset used is inspired by a Kaggle dataset titled **BigMart Sales Prediction** and includes the following:

- **Number of rows**: ~8500
- **Features**:
  - `Item_Weight`
  - `Item_Fat_Content`
  - `Item_Visibility`
  - `Item_Type`
  - `Outlet_Identifier`
  - `Outlet_Establishment_Year`
  - `Outlet_Size`
  - `Outlet_Location_Type`
  - `Outlet_Type`
  - ... and more.

---

## 🔄 Workflow

### 1. **Exploratory Data Analysis (EDA)**
- Univariate and bivariate analysis
- Handling missing values
- Feature correlation
- Visualizations with seaborn/matplotlib

### 2. **Data Preprocessing**
- Handling categorical variables (Label Encoding & One-Hot Encoding)
- Imputation of missing values
- Feature scaling (if required)
- Log transformation for skewed features

### 3. **Model Building**
- Splitting dataset into training and test sets
- Models used:
  - **Linear Regression**
  - **Ridge Regression**
  - **Lasso Regression**
  - **Random Forest Regressor**
  - **XGBoost Regressor**
- Hyperparameter tuning using `GridSearchCV`

### 4. **Model Evaluation**
- Evaluation Metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- Model comparison using a metrics summary table

### 5. **Final Model Deployment (Optional)**
- Saving the best model using `joblib` or `pickle` (optional)
- Predicting on new test data

---

## 📊 Results Summary

| Model                  | MAE     | RMSE    | R² Score |
|------------------------|---------|---------|----------|
| Linear Regression      |  ...    |   ...   |   ...    |
| Random Forest Regressor|  ...    |   ...   |   ...    |
| XGBoost Regressor      |  ...    |   ...   |   ...    |

_(Add actual values from your notebook after running the models)_

---

## 📎 Requirements

To run this project, install the following dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## 🚀 How to Run

1. Clone the repository or download the notebook.
2. Install the dependencies.
3. Run `P1_Sales_Price_prediction_Regression.ipynb` in Jupyter Notebook.
4. Follow each cell in sequence for EDA, preprocessing, modeling, and evaluation.

---

## 📁 Project Structure

```
Sales_Price_Prediction/
│
├── P1_Sales_Price_prediction_Regression.ipynb  # Main notebook
├── dataset/                                    # Contains train/test CSV files
├── model/                                      # (Optional) Saved trained model
└── README.md
```

---

## ✅ Future Enhancements

- Add deployment using Streamlit or Flask for interactive prediction
- Incorporate feature selection techniques
- Apply advanced ensemble techniques (e.g., Stacking)
- Hyperparameter optimization with Optuna/Bayesian Search

---

## 👨‍💻 Author

**Deepak P.**  
Transitioning into AI/ML roles with domain experience in drone-based solar analytics.  
🔗 [LinkedIn](#) | 🌐 Portfolio (optional) | 📧 deepak@email.com
