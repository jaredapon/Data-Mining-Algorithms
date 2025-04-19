import os
os.environ["OMP_NUM_THREADS"] = '1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the Pokémon dataset
df = pd.read_csv("PokemonData.csv")

# Extract 'Base Attack' and 'Base Defense' attributes and Pokémon names
X = df[['Base Attack', 'Base Defense']].values
names = df['Name'].values

# Set the number of clusters
k = 2

# Set random seed for reproducibility
np.random.seed(42)

# Randomly initialize centroids by selecting k data points
initial_indices = np.random.choice(range(X.shape[0]), k, replace=False)
centroids = X[initial_indices]

# Parameters for the algorithm
max_iter = 10   # Maximum iterations allowed
tol = 1e-4      # Tolerance for convergence

print("########## K-MEANS ##########\n")

# K-Means Iteration Loop
for iteration in range(1, max_iter + 1):
    # Compute distances from each point to each centroid (Euclidean)
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    
    # Assign clusters based on closest centroid
    labels = np.argmin(distances, axis=1)
    
    # Compute new centroids as the mean of points in each cluster
    new_centroids = np.array([
        X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
        for j in range(k)
    ])
    
    # Print iteration header and centers
    print(f"=== K-MEANS - Iteration {iteration} ===")
    print("Centers:")
    for j in range(k):
        print(f"  Cluster {j+1} center: {centroids[j]}")
    print("\nPoint Details:")
    
    # Print details for each point: name, coordinates, distances, and assignment
    for idx, point in enumerate(X):
        # Calculate distances to current centers with 2-decimal precision
        dists = [np.linalg.norm(point - centroids[j]) for j in range(k)]
        dists_str = ", ".join(f"{d:.2f}" for d in dists)
        assigned_cluster = labels[idx] + 1  # human-readable (1-indexed)
        print(f"  {names[idx]} {point} -> Distances: [{dists_str}] -> Assigned to Cluster {assigned_cluster}")
    
    print("\n" + "-" * 50 + "\n")
    
    # If this is not the last iteration, print the updated centroids info
    if not np.allclose(centroids, new_centroids, atol=tol):
        print(f"Updated Centroids from Iteration {iteration} to Iteration {iteration+1}:")
        for j in range(k):
            print(f"  Cluster {j+1} new centroid: {new_centroids[j]}")
        print("\n" + "=" * 50 + "\n")
    else:
        print("K-MEANS convergence reached at iteration", iteration)
        break

    # Plot the current iteration (using new_centroids for display)
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue']
    for j in range(k):
        cluster_points = X[labels == j]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=colors[j], s=80, label=f'Cluster {j+1}')
    plt.scatter(new_centroids[:, 0], new_centroids[:, 1],
                color='black', marker='X', s=200, label='Centroids')
    plt.title(f"K-Means Clustering - Iteration {iteration}")
    plt.xlabel("Base Attack")
    plt.ylabel("Base Defense")
    plt.legend()
    plt.grid(True)
    plt.show()

    centroids = new_centroids

# If convergence occurred before max_iter, assign the final labels
labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

# Display the final cluster assignments
print("Final K-MEANS Cluster Assignments:")
for j in range(k):
    cluster_members = [names[i] for i in range(len(names)) if labels[i] == j]
    print(f"  Cluster {j+1}: {cluster_members}")

# Add the final cluster labels to the dataframe and print
df.loc[:, 'KMeans Cluster No'] = labels
cluster_results = df[['Name', 'KMeans Cluster No']].sort_values(by='KMeans Cluster No').reset_index(drop=True)
print("\nFinal Cluster Table:")
print(cluster_results)
