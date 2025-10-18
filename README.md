# Credit-Card-Fraud-Detection

Machine learning project detecting fraudulent credit card transactions using Logistic Regression, Random Forest, and XGBoost

---

## Project Overview

Financial fraud detection is a critical challenge where false negatives (missed frauds) are extremely costly, while false positives can burden customers and investigators.  
This project focuses on **building robust classifiers** and **evaluating them using appropriate metrics** to balance these risks.

**Key objectives:**
1. Explore and visualize transaction data to understand patterns.
2. Handle class imbalance using SMOTE and class weighting.
3. Train and evaluate Logistic Regression, Random Forest, and XGBoost models.
4. Compare models using **Precision–Recall AUC** and confusion matrices.

---

## Dataset

- **Source:** Machine Learning Group (ULB) – [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Rows:** 284,807  
- **Frauds:** 492 (0.17%)  
- **Features:** 30 (PCA-anonymized variables, `Time`, `Amount`, `Class`)

---

## Methodology

### 1. Preprocessing
- Scaled numerical columns (`Time`, `Amount`) using `StandardScaler`
- Kept PCA-transformed features (`V1`–`V28`) unchanged
- Stratified train/test split (80/20)

### 2. Class Imbalance Handling
- Compared three strategies:
  - Baseline models (no resampling)
  - **SMOTE** oversampling
  - Class weighting (`class_weight` for RandomForest, `scale_pos_weight` for XGBoost)

### 3. Models Trained
| Model | Imbalance Handling | Highlights |
|--------|--------------------|-------------|
| Logistic Regression | Baseline + SMOTE | Simple, interpretable, improved recall |
| Random Forest | Class weights | Good balance between precision & recall |
| XGBoost | Scale weights | Strongest overall model |

---

## Results

| Model | Precision | Recall | F1 | PR AUC |
|--------|-----------|--------|----|--------|
| Logistic Regression | 0.80 | 0.68 | 0.73 | 0.74 |
| Logistic + SMOTE | 0.71 | 0.81 | 0.76 | 0.78 |
| Random Forest | 0.96 | 0.75 | 0.84 | 0.85 |
| XGBoost | 0.88 | 0.83 | 0.85 | **0.88** |

- **Best performer:** XGBoost (Precision–Recall AUC: 0.88)
- Cross-validation (5 folds) confirmed strong model stability (CV AP ≈ 0.855)

---

## Key Insights
- Fraud cases are extremely rare, highlighting the need for metrics beyond accuracy.  
- Oversampling (SMOTE) boosts recall but may reduce precision.  
- Ensemble models (Random Forest, XGBoost) achieve the best trade-off.  
- Precision–Recall AUC is the most meaningful metric for this type of problem.  

---

## Next Steps
- Tune decision thresholds to optimize for business risk.
- Experiment with deep learning (Autoencoders, Isolation Forest).
- Deploy a Streamlit dashboard for real-time fraud detection demo.
- Automate retraining with recent transaction data.

---

## Tech Stack

**Languages & Libraries:**
- Python  
- pandas, numpy, matplotlib, seaborn  
- scikit-learn  
- imbalanced-learn (SMOTE)  
- XGBoost  
- joblib  

