# Assignment 2 — Random Forest (From Scratch) & XGBoost Classification

**Course:** Machine Learning (IIT Kharagpur, 2025) · **Roll No:** 22CS30009

Two tree-ensemble classification tasks: a Random Forest built **entirely from scratch**, and a tuned XGBoost pipeline on a real-world churn dataset.

---

## Part 1 — Random Forest From Scratch (Pima Indians Diabetes)

Diabetes prediction from diagnostic measurements (Glucose, BMI, Age, Insulin, etc.).

**Implemented from scratch (no sklearn estimators):**
- `Node` class — decision/leaf node representation
- `DecisionTree._grow_tree` — recursive tree growth with random feature subsets
- `_gini` / `_information_gain` — Gini impurity & split quality
- `RandomForest.fit` — bootstrap sampling (with replacement) per tree
- `RandomForest.predict` — majority vote across trees
- `predict_probability` — fraction of trees voting class 1 (used for ROC)

**Hyperparameters** (per roll-number rule, 22CS30009 → Set 1):
`n_trees = 9`, `max_depth = 7`, `num_features = 8`, `min_samples_split = 2`

**Results:**
- **Train Accuracy: 91.53%**
- Includes confusion matrix, ROC curve, and an interpretable ASCII tree visualization
- Top-level splits align with medical intuition (Glucose, BMI, Age)

---

## Part 2 — XGBoost (Telco Customer Churn)

Predict whether a customer churns, on the IBM Telco dataset (7,043 customers, 33 raw attributes).

**Pipeline:**
- Data cleaning — dropped exit-interview leakage columns (`Churn Label/Score/Reason`, `CLTV`), single-value columns, `CustomerID`
- Missing-value handling — 11 blank `Total_Charges` rows set to 0 (matched `Tenure_Months = 0`)
- One-hot encoding of categoricals
- **Stratified** 75/25 split preserving the 26.5% churn rate in both sets
- `XGBClassifier` (binary:logistic, AUC metric) + `RandomizedSearchCV` hyperparameter tuning
- AUC curves, confusion matrix, ROC, classification report, single-tree feature importance

**Results:**

| Model | Accuracy | Churn (class 1) F1 |
|---|---|---|
| Baseline XGBoost | 0.79 | 0.58 |
| **Tuned (RandomizedSearchCV)** | **0.80** | **0.59** (recall 0.53 → 0.53, precision 0.63 → 0.65) |

Best params: `max_depth=3`, `learning_rate=0.05`, `n_estimators=200`, `subsample=0.6`, `colsample_bytree=0.6`, `min_child_weight=5`, `gamma=0.2`

Most important churn drivers: `Contract_Month-to-month`, `Tech_Support_No`, `Tenure_Months`.

---

## Files

| File | Description |
|---|---|
| `22CS30009_P1.ipynb` | Random Forest from scratch (solved notebook) |
| `22CS30009_P2.ipynb` | XGBoost churn classification (solved notebook) |
| `22CS30009_P1.csv` | Test-set predictions (Random Forest) |
| `22CS30009_test_prob.npy` | Predicted class-1 probabilities for ROC evaluation |
| `ML Assignment 2.pdf` | Original assignment statement |
