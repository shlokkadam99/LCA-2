# LCA-2
# Breast Cancer Classification using KNN

A simple machine learning project that uses the K-Nearest Neighbors (KNN)
algorithm to classify breast tumors as **malignant** or **benign**.

## Overview

Breast cancer diagnosis relies on measurements taken from a fine needle
aspirate (FNA) of a breast mass. This project trains a KNN classifier on
those measurements to predict whether a tumor is malignant or benign, and
reports how accurate the model is on unseen data.

## Dataset

- **Breast Cancer Wisconsin (Diagnostic) Data Set** from the UCI Machine
  Learning Repository.
- 569 samples, 30 numeric features (radius, texture, perimeter, area,
  smoothness, concavity, symmetry, etc.).
- 2 classes: malignant (212 samples) and benign (357 samples).
- No missing values.
- Loaded directly via `sklearn.datasets.load_breast_cancer()` — no manual
  download required.

## How it works

1. Load the dataset.
2. Split into training (80%) and testing (20%) sets.
3. Scale the features using `StandardScaler`, since KNN is distance-based
   and sensitive to feature magnitude.
4. Train a `KNeighborsClassifier` (k = 5).
5. Predict on the test set.
6. Evaluate using accuracy and a classification report (precision, recall,
   F1-score).

## Project structure

```
breast-cancer-knn/
├── knn_classifier.py     # Main script
├── requirements.txt      # Dependencies
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/breast-cancer-knn.git
cd breast-cancer-knn
pip install -r requirements.txt
```

## Run

```bash
python knn_classifier.py
```

## Output

```
Accuracy: 95.61 %
              precision    recall  f1-score   support

   malignant       0.95      0.93      0.94        42
      benign       0.96      0.97      0.97        72

    accuracy                           0.96       114
```

## Why KNN?

KNN classifies a data point based on the majority class among its `k`
nearest neighbors. It's a good fit here because the features are continuous,
scale well with standardization, and the two classes are reasonably
well-separated.

## Possible improvements

- Tune `k` using cross-validation.
- Compare with other classifiers (SVM, Logistic Regression, Random Forest).
- Add a confusion matrix plot.

## Tech Stack

- Python 3
- scikit-learn


