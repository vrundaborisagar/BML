# Experiment 12 - Performance Matrix / Evaluation

## Overview
This experiment focuses on **evaluating a classification model** using performance metrics and visualizations. It employs the **Gradient Boosting Classifier** to predict wine quality and evaluates the model using accuracy, classification report, confusion matrix, and a bar plot of predicted class distribution.

---

## Dataset Information
- **Dataset**: `winequality-red.csv`
- **Separator**: `;` (semicolon)
- **Target Variable**: `quality`
- The target labels are shifted using `y = y - 3` to start from 0.

---

## Libraries Used
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
```
## code
```python
# Load the dataset
df = pd.read_csv("/content/drive/MyDrive/winequality-red.csv", sep=';')

# Define features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Shift target labels to start from 0
y = y - 3

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions
y_pred = gb_model.predict(X_test)

# Evaluation metrics
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Bar plot of predicted wine quality
predicted_quality = pd.Series(y_pred)
quality_counts = predicted_quality.value_counts().sort_index()

plt.figure(figsize=(8,6))
sns.barplot(x=quality_counts.index, y=quality_counts.values, palette="viridis")
plt.xlabel('Predicted Wine Quality (After Shifting)')
plt.ylabel('Number of Wines')
plt.title('Distribution of Predicted Wine Quality')
plt.grid(axis='y')
plt.show()
```
## output
```python
Accuracy:  0.65

Classification Report:
               precision    recall  f1-score   support

           0       0.00      0.00      0.00         1
           1       0.50      0.10      0.17        10
           2       0.70      0.77      0.74       130
           3       0.63      0.65      0.64       132
           4       0.60      0.50      0.55        42
           5       0.00      0.00      0.00         5

    accuracy                           0.65       320
   macro avg       0.41      0.34      0.35       320
weighted avg       0.64      0.65      0.64       320
```
![download (1),png](https://github.com/user-attachments/assets/f6ed9782-c934-4cb2-8593-bdf6e50e834c)

![download](https://github.com/user-attachments/assets/55666559-1617-411f-8ee3-d93068497530)
```

