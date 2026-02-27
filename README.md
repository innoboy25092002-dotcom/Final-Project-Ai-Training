# 🏦 Bank Marketing Response Prediction

> Predicting whether a client will subscribe to a term deposit using machine learning on the Bank Marketing Dataset.

---

## 📌 Problem Statement

Banks run marketing campaigns to promote term deposit subscriptions. Calling every client is expensive and inefficient. This project builds a **binary classification model** that predicts whether a client will subscribe (`yes`) or not (`no`) to a term deposit — helping the bank focus its efforts on clients most likely to convert.

---

## 📂 Dataset

| Property | Details |
|----------|---------|
| Source | [Kaggle – Bank Marketing Dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset) |
| Records | 11,162 |
| Features | 16 input features + 1 target |
| Target | `deposit` → `yes` / `no` |
| Class Balance | ~53% No, ~47% Yes (near balanced) |

### Feature Overview

| Feature | Description |
|---------|-------------|
| `age` | Client's age |
| `job` | Type of job |
| `marital` | Marital status |
| `education` | Education level |
| `default` | Has credit in default? |
| `balance` | Average yearly balance (EUR) |
| `housing` | Has housing loan? |
| `loan` | Has personal loan? |
| `contact` | Contact communication type |
| `day` | Last contact day of month |
| `month` | Last contact month |
| `duration` | Last contact duration (seconds) |
| `campaign` | Number of contacts during campaign |
| `pdays` | Days since last contact from previous campaign |
| `previous` | Number of contacts before this campaign |
| `poutcome` | Outcome of previous marketing campaign |

---

## 🤖 Model

**Algorithm:** Gradient Boosting Classifier

Gradient Boosting was chosen for its ability to handle mixed data types, capture non-linear relationships, and provide reliable feature importance scores — all critical for this type of banking dataset.

### Hyperparameters

```python
GradientBoostingClassifier(
    n_estimators   = 300,
    learning_rate  = 0.07,
    max_depth      = 4,
    subsample      = 0.85,
    min_samples_leaf = 20,
    random_state   = 7
)
```

---

## ⚙️ Feature Engineering

Three new features were crafted from existing columns:

| New Feature | How it's built | Why |
|-------------|----------------|-----|
| `age_bucket` | Age grouped into 4 life stages (0–30, 31–40, 41–55, 55+) | Captures non-linear age effects |
| `call_efficiency` | `duration / (campaign + 1)` | Measures engagement per contact |
| `was_contacted_before` | Binary flag: `pdays != -1` | Separates new vs. returning leads |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 86.07% |
| **ROC-AUC** | 0.9266 |
| **5-Fold CV AUC** | 0.9265 ± 0.0055 |

### Classification Report

```
              precision    recall  f1-score   support

  No Deposit       0.89      0.84      0.86      1175
  Subscribed       0.83      0.88      0.86      1058

    accuracy                           0.86      2233
   macro avg       0.86      0.86      0.86      2233
weighted avg       0.86      0.86      0.86      2233
```

### Model Visualizations

![Model Results](bank_model_results.png)

*Includes: Confusion Matrix · ROC Curve · Feature Importances · Probability Distribution · Cross-Validation Scores*

---

## 🗂️ Project Structure

```
bank-marketing-prediction/
│
├── bank.csv                    # Dataset
├── bank_model.py               # Main model script
├── bank_model_results.png      # Output visualizations
└── README.md                   # This file
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/bank-marketing-prediction.git
cd bank-marketing-prediction
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib
```

### 3. Run the Model

```bash
python bank_model.py
```

This will:
- Load and preprocess the dataset
- Engineer new features
- Train the Gradient Boosting model
- Print evaluation metrics
- Save `bank_model_results.png` with all visualizations
- Show sample predictions with confidence scores

---

## 📦 Dependencies

| Library | Version |
|---------|---------|
| Python | 3.8+ |
| pandas | ≥ 1.3 |
| numpy | ≥ 1.21 |
| scikit-learn | ≥ 1.0 |
| matplotlib | ≥ 3.4 |

---

## 🔍 Key Findings

- **`duration`** (call duration) is the strongest predictor — longer calls strongly correlate with subscription
- **`call_efficiency`** (engineered feature) ranks in the top 5 most important features
- **`balance`** and **`poutcome`** (previous campaign outcome) are also highly influential
- The model generalizes well: CV AUC (0.9265) closely matches test AUC (0.9266), indicating no overfitting

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋 Acknowledgements

- Dataset sourced from [Kaggle](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)
- Original data from the UCI Machine Learning Repository
