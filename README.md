# 📊 Tabular ML Classification Study

## 📌 Overview

This project presents a **research-oriented machine learning pipeline** for solving a binary classification problem on high-dimensional tabular data.

The goal is to build a structured workflow that prepares for advanced topics such as **Deep Learning, Graph Neural Networks, and optimization techniques**.

---

## 🎯 Objectives

* Develop a complete classification pipeline
* Handle real-world challenges such as:

  * Class imbalance
  * High-dimensional data
  * Feature engineering without domain knowledge
* Compare multiple machine learning models
* Apply research-driven experimentation methodology

---

## 📁 Project Structure

```
Classification-Comparison/

data/
    train.csv

notebooks/
    00_gpu_check.ipynb
    01_classification_baseline.ipynb
    02_preprocessing_experiments.ipynb
    03_feature_engineering.ipynb
    04_encoding_feature_selection.ipynb
    05_model_tuning.ipynb
    06_final_model.ipynb

models/
    final_model.pkl
    features.pkl
    threshold.pkl

reports/
    results_summary.md
```

---

## 📊 Dataset

* High-dimensional tabular dataset (~200 features)
* Binary target variable:

  * `0` → No transaction
  * `1` → Transaction
* Highly imbalanced:

  * ~90% class 0
  * ~10% class 1

---

## 🔬 Methodology

### 1. Baseline Models

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

👉 Observation: High accuracy but poor minority class detection

---

### 2. Handling Class Imbalance

* Applied `class_weight='balanced'`
* Performed threshold tuning

👉 Improved F1-score significantly

---

### 3. Feature Engineering

* Created statistical features:

  * Mean
  * Standard deviation
  * Min / Max
  * Range

👉 Improved both F1-score and ROC-AUC

---

### 4. Feature Selection

* Used Random Forest importance
* Selected top features

👉 Result: Performance decreased (important insight)

---

### 5. Advanced Models

* XGBoost (GPU-enabled)

👉 Achieved best performance

---

### 6. Hyperparameter Tuning

* Applied RandomizedSearchCV

👉 Insight: Tuning did not outperform default model

---

## 📈 Final Results

| Model               | F1-score  | ROC-AUC   |
| ------------------- | --------- | --------- |
| Logistic Regression | ~0.47     | ~0.864    |
| Random Forest       | 0.0       | ~0.826    |
| XGBoost (default)   | **~0.51** | **~0.87** |

---

## 🏆 Final Model

**XGBoost (default configuration with GPU acceleration)**

---

## 🔥 Key Insights

* Feature engineering has the highest impact on performance
* Handling class imbalance is critical
* ROC-AUC is the most reliable metric for imbalanced data
* Default models can outperform tuned versions
* Tree boosting models outperform bagging methods

---

## 💾 Model Saving

Saved artifacts:

* `final_model.pkl`
* `features.pkl`
* `threshold.pkl`

---

## 🚀 Future Work

* Deep Learning for tabular data
* Optimization techniques
* Graph Neural Networks (GNNs)

---

## 🧠 Learning Outcome

This project demonstrates:

* End-to-end ML pipeline development
* Research-oriented experimentation
* Strong foundation for advanced ML and DL research

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost (GPU)
* Jupyter Notebook

---

## 👨‍💻 Author

**Ibad Rehman**

---
