import numpy as np


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Function to do  K-means 
def kmeans(data_points, k=4, num_iterations=100):
    # Randomly choose centroids
    centroids = data_points[np.random.choice(range(len(data_points)), k, replace=False)]

    for _ in range(num_iterations):
        # Assign clusters
        clusters = {}
        for x in data_points:
            distances = [euclidean_distance(x, centroid) for centroid in centroids]
            cluster_index = distances.index(min(distances))
            if cluster_index not in clusters:
                clusters[cluster_index] = []
            clusters[cluster_index].append(x)

        # Update centroids
        new_centroids = []
        for cluster_index in sorted(clusters):
            new_centroids.append(np.mean(clusters[cluster_index], axis=0))

        centroids = np.array(new_centroids)

    wcss = sum(
        sum(euclidean_distance(x, centroids[cluster_index])**2 for x in clusters[cluster_index])
        for cluster_index in clusters
    )

    return centroids, wcss

epsilon = 0.01

data_points = np.array([([np.cos(m * np.pi / 2 + epsilon), np.sin(m * np.pi / 2 + epsilon)],
                        [np.cos(m * np.pi / 2 - epsilon), np.sin(m * np.pi / 2 - epsilon)])
                        for m in range(1, 5)]).reshape(-1, 2)

best_wcss = np.inf
best_centroids = None

for _ in range(10):
    centroids, wcss = kmeans(data_points)
    if wcss < best_wcss:
        best_wcss = wcss
        best_centroids = centroids


print("Best WCSS:", best_wcss)

