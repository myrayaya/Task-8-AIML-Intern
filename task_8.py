# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# importing the dataset
df = pd.read_csv('dataset/Mall_Customers.csv')
print(df.head())

# selecting numerical columns
x = df.select_dtypes(include = [np.number])

# data visualization (pairplot and distribution)
sns.pairplot(x)
plt.show()

# elbow method to find optimal K
inertia = []
K =range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters = k, random_state = 42, n_init = 10)
    kmeans.fit(x)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()

# choosing optimal cluster count
optimal_k = 5
kmeans = KMeans(n_clusters = optimal_k, random_state = 42, n_init = 10)
labels = kmeans.fit_predict(x)

# adding labels to dataframe
x['Cluster'] = labels

# evaluate model using silhouette score
silhouette_avg = silhouette_score(x, labels)
print(f'Silhouette Score for k = {optimal_k}: {silhouette_avg:.3f}')

# visualizing clusters using PCA
pca = PCA(n_components = 2)
x_pca = pca.fit_transform(x)

plt.figure(figsize = (8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c = labels, cmap = 'viridis', s = 50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 200, c = 'red', marker = 'X', label = 'Centroids')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering Visualization (PCA-reduced)')
plt.legend()
plt.show()