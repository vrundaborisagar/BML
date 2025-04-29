# Experiment 11 - Boosting

## Overview
**Boosting** is a sequential ensemble technique that builds models iteratively. Each new model focuses on correcting the errors made by the previous ones, improving the overall performance over time.

This experiment uses the **Wine Quality** dataset and implements **XGBoost** (Extreme Gradient Boosting) for multi-class classification.

---

## Dataset Information
- **File**: `winequality-red.csv`
- **Separator**: `;` (semicolon)
- **Target Variable**: `quality`
- The target is shifted (`y = y - 3`) to start the class labels from 0.

---

## Libraries Used
```python
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```
```python
# Load the dataset
df = pd.read_csv("/content/drive/MyDrive/winequality-red.csv", sep=';')

# Define features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Shift labels so classes start from 0
y = y - 3  # Minimum quality is 3

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost classifier
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Predict on the test data
y_pred = xgb_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

## output
```python
Accuracy: 0.696875

Classification Report:
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         1
           1       0.00      0.00      0.00        10
           2       0.75      0.80      0.78       130
           3       0.68      0.73      0.70       132
           4       0.64      0.55      0.59        42
           5       0.00      0.00      0.00         5

    accuracy                           0.70       320
   macro avg       0.34      0.35      0.34       320
weighted avg       0.67      0.70      0.68       320

Confusion Matrix:
[[  0   0   1   0   0   0]
 [  0   0   7   3   0   0]
 [  0   1 104  24   1   0]
 [  0   1  25  96   9   1]
 [  0   0   1  17  23   1]
 [  0   0   0   2   3   0]]
```
