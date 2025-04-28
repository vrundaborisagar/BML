# K-Nearest Neighbors (KNN) Classifier

## Overview
This project implements a K-Nearest Neighbors (KNN) classifier using Python. KNN is a simple, non-parametric algorithm used for classification and regression tasks. It classifies a data point based on the majority class of its nearest neighbors.

## Features
- Implementation using `scikit-learn`
- Training and testing a KNN classifier
- Performance evaluation using accuracy
- Adjustable `k` value for tuning

## Requirements
Ensure you have the following Python libraries installed:

```sh
pip install numpy pandas scikit-learn matplotlib
```

## Usage

### Run the Script
Execute the Python script to train and test the KNN model:

```sh
python knn_classifier.py
```

## Example Code
```python
import csv
import random
import math

# Load dataset
def load_csv(filename):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row:
                continue
            # Convert features to float, keep label (target) as string
            data.append([float(x) for x in row[:-1]] + [row[-1]])
    return data

# Split dataset
def train_test_split(data, test_size=0.2, random_state=42):
    random.seed(random_state)
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    return data[:split_index], data[split_index:]

# Euclidean distance
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)

# Get nearest neighbors
def get_neighbors(train, test_row, k):
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

# Predict
def predict_classification(train, test_row, k):
    neighbors = get_neighbors(train, test_row, k)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

# Accuracy
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Main
filename = '/content/drive/MyDrive/Datasets/iris.data'
dataset = load_csv(filename)

train_data, test_data = train_test_split(dataset, test_size=0.2)

k = 3

predictions = []
for row in test_data:
    prediction = predict_classification(train_data, row, k)
    predictions.append(prediction)

actual = [row[-1] for row in test_data]

acc = accuracy_metric(actual, predictions)
print(f"Accuracy: {acc:.2f}%")

```
### Output:
```
KNN model Accuracy: 96.67%

```


## License
This project is licensed under the MIT License.

