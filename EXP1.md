# EXTRACT THE DATA FROM DATABASE USING PYTHON
## Introduction
 This program analyzes the Iris dataset using the pandas library. It begins by loading the dataset and displaying key information, such as the first few rows, column details, and dataset size. The program also checks for missing values, handles them by filling with zeros, and provides descriptive statistics to understand the dataset better. This script is useful for basic data exploration and preprocessing before further analysis or machine learning applications.
<br>

## overview
This program performs exploratory data analysis (EDA) on the Iris dataset using pandas. It follows these steps:

Load the dataset – Reads the Iris dataset from a CSV file.
<br>
Display data preview – Shows the first five rows.
<br>
Dataset information – Prints column names, data types, and memory usage.
<br>
Statistical summary – Generates descriptive statistics for numerical columns.
<br>
Check dataset size and shape – Displays the total number of elements and dimensions.
<br>
Handle missing values – Identifies missing data, displays affected rows (if any), and fills missing values with zero.
<br>
Final verification – Ensures all missing values are handled.

##code
```
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
iris_data = pd.read_csv('/content/drive/MyDrive/iris.data')

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(iris_data.head())

# Get dataset information (columns, non-null counts, datatypes, memory usage)
print("\nDataset Information:")
print(iris_data.info())

# Get the total number of elements in the DataFrame
print("\nTotal number of elements in the dataset:")
print(iris_data.size)

# Generate descriptive statistics for numeric columns
print("\nDescriptive statistics:")
print(iris_data.describe())

# Get the shape of the dataset (rows, columns)
print("\nShape of the dataset:")
print(iris_data.shape)

# Check for missing values in each column
print("\nMissing values in each column:")
print(iris_data.isnull().sum())

# Check if there are any missing values in the entire dataset
print("\nAre there any missing values in the dataset?")
print(iris_data.isnull().any().any())

# Display rows with missing values, if any
missing_rows = iris_data[iris_data.isnull().any(axis=1)]
if missing_rows.empty:
    print("\nNo rows contain missing values.")
else:
    print("\nRows with missing values:")
    print(missing_rows)

# Optionally, fill missing values with 0
iris_data.fillna(value=0, inplace=True)
print("\nMissing values have been filled with 0.")

# Verify again after filling missing values
print("\nMissing values check after filling:")
print(iris_data.isnull().sum())

# Assign column names if they are missing
iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Display the first few rows
print(iris_data.head())

plt.figure(figsize=(8, 6))
iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].boxplot()
plt.title('Box Plot of Iris Features')
plt.ylabel('Measurement (cm)')
plt.grid(True)
plt.show()

