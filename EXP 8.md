# K-Means Clustering on Iris Dataset

This project applies **K-Means Clustering** on the famous **Iris dataset** using two features: *sepal length* and *sepal width*.  
The dataset is first **standardized**, then **clustered** into 3 groups, and finally **visualized**.

---

## Project Structure

- **Load Dataset**  
  Load the Iris dataset and assign appropriate column names.

- **Feature Selection**  
  Select *sepal length* and *sepal width* as the features for clustering.

- **Data Preprocessing**  
  Standardize features using `StandardScaler` to improve clustering performance.

- **K-Means Clustering**  
  Apply K-Means algorithm to group the data points into 3 clusters.

- **Visualization**  
  Plot the clusters along with the centroids.

---

## Dependencies

Make sure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

You can install them using:

```bash
pip install numpy pandas matplotlib scikit-learn
```
## code
```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset
# Assign column names to the Iris dataset
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']
df = pd.read_csv('/content/drive/MyDrive/Datasets/iris.data', names=columns)

# 2. Feature Selection
# Select only sepal length and sepal width for clustering
X = df[['sepal_length', 'sepal_width']]

# 3. Data Preprocessing
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. K-Means Clustering
# Define the number of clusters (k)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# 5. Adding Cluster Labels
# Add the cluster labels to the original dataframe
df['Cluster'] = kmeans.labels_

# 6. Visualization
# Plotting the clusters and centroids
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis', edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='red', marker='X', label='Centroids')
plt.title('K-Means Clustering on Iris Dataset')
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.legend()
plt.grid(True)
plt.show()

```
## output
![image](https://github.com/user-attachments/assets/fb1adc4c-8c44-4e53-bb59-c4a69fb58fdc)

## Conclusion

- The **K-Means algorithm** successfully grouped the Iris dataset into **three distinct clusters**.
- The clusters roughly correspond to the natural species categories: *Iris-setosa*, *Iris-versicolor*, and *Iris-virginica*.
- **Standardizing** the features helped ensure that both dimensions contributed equally during clustering.
- Although the clustering worked well for the chosen features (*sepal length* and *sepal width*), some **overlap** between species can still exist due to feature similarities.
- The visualization shows a clear separation of clusters with centroids highlighted in red.

