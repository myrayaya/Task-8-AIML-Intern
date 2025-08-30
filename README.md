# Task-8: K-Means Clustering

## Objective
Implement **unsupervsied learning** using **K-Means clustering** and evaluate the clustering performance.

## Dataset
- Used Mall Customers dataset from Kaggle.com

## Steps Performed
1. **Loading Dataset**: imported the csv file using Pandas
2. **Preprocessing**: Selected numerical columns for clustering
3. **Elbow Method**: Plotted inertia vs k to find the optimal cluster count
4. **K-Means**: Fitted a K-Means model, assigned cluster labels
5. **Evaluation**: Used **Silhouette Score** to evaluate clustering quality
6. **Visualization**: Plotted clustered data using **PCA (2D projection)**

# Results
- Optimal Number of Clusters: ~5 (based on Elbow and Silhouette)
- Silhouette Score : 0.426
- Cluster Visualization shows well-seperated groups
<img width="878" height="193" alt="image" src="https://github.com/user-attachments/assets/c2b8c423-fa78-4433-b988-6e54a96bdd9e" />
<img width="1852" height="932" alt="image" src="https://github.com/user-attachments/assets/597d2622-bc2d-4767-a78a-427bbf5063de" />
<img width="1063" height="573" alt="image" src="https://github.com/user-attachments/assets/a69de336-04af-428e-9312-d2f8a1ef6b2a" />
<img width="887" height="683" alt="image" src="https://github.com/user-attachments/assets/f13926a9-82ce-4524-94f8-ba2d1989aee3" />

## Author
- Myra Chauhan
