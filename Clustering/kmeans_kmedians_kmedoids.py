import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances

# Load the Pokémon dataset
df = pd.read_csv(r"C:\Users\Jared Apon\Desktop\PokemonData.csv")

# Extract 'Base Attack' and 'Base Defense' attributes and Pokémon names
X = df[['Base Attack', 'Base Defense']].values
names = df['Name'].values

# Set the number of clusters
k = 5

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
    colors = ['red', 'blue', 'purple', 'pink', 'orange']
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

# Final iteration plot
plt.figure(figsize=(8, 6))
for j in range(k):
    cluster_points = X[labels == j]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                color=colors[j], s=80, label=f'Cluster {j+1}')
plt.scatter(centroids[:, 0], centroids[:, 1],
            color='black', marker='X', s=200, label='Centroids')
plt.title("Final K-Means Clustering on Pokémon (Base Attack vs. Base Defense)")
plt.xlabel("Base Attack")
plt.ylabel("Base Defense")
plt.legend()
plt.grid(True)
plt.show()


##########################################################################
# Load the Pokémon dataset
df = pd.read_csv(r"C:\Users\Jared Apon\Desktop\PokemonData1.csv")
df_subset = df.copy()

# Extract 'Base Attack', 'Base Defense' attributes and Pokémon names
X = df_subset[['Base Attack', 'Base Defense']].values
names = df_subset['Name'].values

# Set the number of clusters
k = 5

# Initialize medians randomly from the data points
np.random.seed(42)
initial_indices = np.random.choice(len(X), k, replace=False)
medians = X[initial_indices]

print("\n########## K-MEDIANS ##########\n")

max_iters = 100

for iteration in range(1, max_iters + 1):
    # Assign clusters based on Manhattan (cityblock) distance
    labels = np.argmin(cdist(X, medians, metric='cityblock'), axis=1)
    
    # Compute new medians for each cluster
    new_medians = np.zeros_like(medians, dtype=float)
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            new_medians[i] = np.median(cluster_points, axis=0)
        else:
            new_medians[i] = medians[i]
    
    # Print iteration header and centers
    print(f"=== K-MEDIANS - Iteration {iteration} ===")
    print("Centers:")
    for i in range(k):
        center_str = np.array2string(medians[i], separator=' ', formatter={'float_kind':lambda x: f"{x:.0f}"})
        print(f"  Cluster {i+1} center: {center_str}")
    
    print("\nPoint Details:")
    for idx, point in enumerate(X):
        # Compute Manhattan distances to current medians
        dists = cdist([point], medians, metric='cityblock')[0]
        dists_str = ", ".join(f"{d:.2f}" for d in dists)
        assigned_cluster = np.argmin(dists) + 1
        point_str = np.array2string(point, separator=' ', formatter={'float_kind':lambda x: f"{x:.0f}"})
        print(f"  {names[idx]} {point_str} -> Distances: [{dists_str}] -> Assigned to Cluster {assigned_cluster}")
    
    print("\n" + "-" * 50 + "\n")
    
    # Check for convergence
    if np.allclose(medians, new_medians):
        print(f"K-MEDIANS convergence reached at iteration {iteration}")
        break
    else:
        print(f"Updated Medians from Iteration {iteration} to Iteration {iteration+1}:")
        for i in range(k):
            new_center_str = np.array2string(new_medians[i], separator=' ', formatter={'float_kind':lambda x: f"{x:.1f}"})
            print(f"  Cluster {i+1} new median: {new_center_str}")
        print("\n" + "=" * 50 + "\n")
    
    # Plotting the current iteration
    plt.figure(figsize=(8, 6))
    for i in range(k):
        cluster_points = X[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=colors[i], s=80, label=f'Cluster {i+1}')
    plt.scatter(new_medians[:, 0], new_medians[:, 1],
                color='black', marker='X', s=200, label='Medians')
    plt.title(f"K-Medians Clustering - Iteration {iteration}")
    plt.xlabel("Base Attack")
    plt.ylabel("Base Defense")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    medians = new_medians.copy()

# Assign final cluster labels to the dataframe
df_subset.loc[:, 'KMedians Cluster No'] = labels

# Plot final clustering result
plt.figure(figsize=(8, 6))
for i in range(k):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                color=colors[i], s=80, label=f'Cluster {i+1}')
plt.scatter(medians[:, 0], medians[:, 1],
            color='black', marker='X', s=200, label='Medians')
plt.title("Final K-Medians Clustering on Pokémon (Base Attack vs. Base Defense)")
plt.xlabel("Base Attack")
plt.ylabel("Base Defense")
plt.legend()
plt.grid(True)
plt.show()

# Display final cluster assignments
print("Final K-MEDIANS Cluster Assignments:")
for i in range(k):
    members = [names[j] for j in range(len(names)) if labels[j] == i]
    print(f"  Cluster {i+1}: {members}")
    
####################################################
# Load the Pokémon dataset
df = pd.read_csv(r"C:\Users\Jared Apon\Desktop\PokemonData1.csv")
df_subset = df.copy()

# Extract 'Base Attack' and 'Base Defense' attributes and Pokémon names
X = df_subset[['Base Attack', 'Base Defense']].values
names = df_subset['Name'].values

# Compute the distance matrix (Euclidean)
distance_matrix = pairwise_distances(X, metric='euclidean')

# Set the number of clusters
k = 3

# PAM algorithm initialization: randomly choose k medoids
np.random.seed(42)
n = X.shape[0]
medoid_indices = np.random.choice(n, k, replace=False)
current_cost = np.sum(np.min(distance_matrix[:, medoid_indices], axis=1))

print("\n########## K-MEDOIDS ##########\n")
print("Initial Medoids:")
for med in medoid_indices:
    print(f"  Index {med}: {names[med]}")
print(f"Initial cost: {current_cost}\n")

max_iter = 100  # Maximum number of outer iterations allowed

# PAM iterative improvement loop
for iteration in range(1, max_iter + 1):
    print("=" * 30)
    print(f"K-MEDOIDS - Outer Iteration {iteration}")
    print("Current Medoids:")
    for med in medoid_indices:
        print(f"  Index {med}: {names[med]}")
    print(f"Current cost: {current_cost}")
    
    best_swap_cost = current_cost
    best_swap = None
    candidate_count = 0

    # Loop through all candidate swaps (each medoid with every non-medoid candidate)
    for medoid in medoid_indices:
        for candidate in range(n):
            if candidate in medoid_indices:
                continue
            candidate_count += 1
            candidate_medoids = medoid_indices.copy()
            # Perform swap: replace the current medoid with candidate
            candidate_medoids[candidate_medoids == medoid] = candidate
            # Calculate total cost for this configuration
            cost_candidate = np.sum(np.min(distance_matrix[:, candidate_medoids], axis=1))
            print(f"  Candidate swap {candidate_count:3d}: Replace medoid {medoid} ({names[medoid]}) with candidate {candidate} ({names[candidate]}) -> cost: {cost_candidate:.2f}")
            if cost_candidate < best_swap_cost:
                best_swap_cost = cost_candidate
                best_swap = (medoid, candidate, candidate_medoids)

    if best_swap is None or best_swap_cost >= current_cost:
        print("No candidate swap improved the cost. K-MEDOIDS converged.\n")
        break
    else:
        medoid_to_replace, candidate, new_medoids = best_swap
        print(f"\n--> Best swap: Replace medoid {medoid_to_replace} ({names[medoid_to_replace]}) with candidate {candidate} ({names[candidate]})")
        print(f"    Cost reduced from {current_cost:.2f} to {best_swap_cost:.2f}.\n")
        # Update medoids with the best swap found
        medoid_indices = new_medoids
        current_cost = best_swap_cost

    # Plot the current clustering state after the accepted swap
    labels = np.argmin(distance_matrix[:, medoid_indices], axis=1)
    plt.figure(figsize=(8, 6))
    for i in range(k):
        cluster_points = X[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=colors[i], s=80, label=f'Cluster {i+1}')
    medoid_points = X[medoid_indices]
    plt.scatter(medoid_points[:, 0], medoid_points[:, 1],
                color='black', marker='X', s=200, label='Medoids')
    plt.title(f"K-Medoids Clustering - After Iteration {iteration}")
    plt.xlabel("Base Attack")
    plt.ylabel("Base Defense")
    plt.legend()
    plt.grid(True)
    plt.show()

# Final cluster assignment after convergence
labels = np.argmin(distance_matrix[:, medoid_indices], axis=1)
df_subset['KMedoids Cluster No'] = labels

# Plot final clustering result
plt.figure(figsize=(8, 6))
for i in range(k):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                color=colors[i], s=80, label=f'Cluster {i+1}')
plt.scatter(X[medoid_indices, 0], X[medoid_indices, 1],
            color='black', marker='X', s=200, label='Medoids')
plt.title("Final K-Mediods Clustering on Pokémon (Base Attack vs. Base Defense)")
plt.xlabel("Base Attack")
plt.ylabel("Base Defense")
plt.legend()
plt.grid(True)
plt.show()

# Display final cluster assignments
print("Final K-Mediods Cluster Assignments:")
for cluster in range(k):
    members = [names[i] for i in range(len(names)) if labels[i] == cluster]
    print(f"  Cluster {cluster+1}: {members}")