# Assignment 1 — Linear Regression, Logistic Regression & SVM (From Scratch)

**Course:** Machine Learning (IIT Kharagpur, 2025) · **Roll No:** 22CS30009

All models in this assignment are implemented **from scratch using NumPy only** — loss functions, gradients, and batch gradient descent optimization.

---

## Part 1 — Linear Regression (House Price Prediction)

A real-estate regression task: predict house price per unit area from input features.

**Implemented from scratch:**
- `load_data` — data loading into `X`, `y` numpy arrays
- `loss_function` — Mean Squared Error cost
- `compute_gradient` — analytic gradients w.r.t. `w` and `b`
- `batch_gradient_descent` — parameter optimization loop
- Min-max scaling + custom train/test split (75/25)

**Results:**

| Learning Rate | Train Error | Test Error |
|---|---|---|
| 0.001 (default) | 23.16 | 14.97 |
| **0.01 (best)** | **12.44** | **9.22** |

---

## Part 2 — Logistic Regression & SVM (Pumpkin Seed Classification)

Binary classification of pumpkin seed types — **Çerçevelik (0)** vs **Ürgüp Sivrisi (1)** — using morphological features (area, perimeter, axis lengths, etc.).

**Implemented from scratch:**
- Sigmoid function, cross-entropy loss, gradient computation
- Batch gradient descent training for logistic regression
- SVM classifier with hyperparameter experimentation
- Learning-rate comparison experiments

**Results:**

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression (α = 0.01) | 50.33% | 50.08% |
| SVM (best config) | **89.27%** | **88.60%** |

> Logistic regression plateaued near chance level on this dataset, while the SVM clearly separates the two seed types.

---

## Files

| File | Description |
|---|---|
| `ml-assignment-1-part1.ipynb` | Linear Regression notebook (solved) |
| `ml-assigment-1-part2.ipynb` | Logistic Regression + SVM notebook (solved) |
| `22CS30009_Linear.csv` | Test-set predictions (Linear Regression) |
| `22CS30009_Logistic.csv` | Test-set predictions (Logistic Regression) |
| `22CS30009_SVM.csv` | Test-set predictions (SVM) |
