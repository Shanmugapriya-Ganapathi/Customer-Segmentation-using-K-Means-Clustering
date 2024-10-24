import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = r'customer-segmmendatio\Mall_Customers.csv' 
data = pd.read_csv(file_path)

# Check the first few rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Check the columns in the dataset
print("\nColumns in the dataset:")
print(data.columns)

# Feature selection: Selecting 'Annual Income (k$)' and 'Spending Score (1-100)' for clustering
selected_features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardizing the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_features)

# Applying K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualizing the Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=data, palette='Set1', s=100)
plt.title('Customer Segmentation Based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)
plt.show()

# Displaying cluster centers
centers = kmeans.cluster_centers_
print("\nCluster Centers:")
print(centers)
