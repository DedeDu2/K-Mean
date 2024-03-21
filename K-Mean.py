from sklearn.cluster import KMeans
import numpy as np

# Generate random data points
np.random.seed(0)
X = np.random.randn(100, 2)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=300, c='red')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
