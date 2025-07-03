from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

num_samples_total = 1000
cluster_centers = [(3, 3), (7, 7)]
num_classes = len(cluster_centers)
epsilon = 0.3
min_samples = 13

# Generate data
X, y = make_blobs(n_samples=num_samples_total,
                  centers=cluster_centers,
                  n_features=num_classes,
                  center_box=(0, 1),
                  cluster_std=0.5)

np.save('./clusters.npy', X)
X = np.load('./clusters.npy')

# Execute DBSCAN
db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
labels = db.labels_

# Remove the noise
range_max = len(X)
X = np.array([X[i] for i in range(0, range_max) if labels[i] != -1])
labels = np.array([labels[i] for i in range(0, range_max) if labels[i] != -1])

def func(x: int):
    if x == 1:
        return '#3b4cc0'

    if x < 0:
        return '#fa0291'
    
    return "#30b404"

# Show clusters
colors = list(map(func, labels))
plt.scatter(X[:, 0], X[:, 1], c=colors, marker="o", picker=True)
plt.title('Two clusters with data')
plt.xlabel('Axis X[0]')
plt.ylabel('Axis X[1]')
plt.savefig("clusters.png")
