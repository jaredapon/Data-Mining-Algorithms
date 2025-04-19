import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

# Set environment variable to avoid potential issues on Windows
os.environ["OMP_NUM_THREADS"] = '1'

# Load the Pokémon dataset
df = pd.read_csv(r"C:\Users\Jared Apon\Desktop\PokemonData.csv")
df_subset = df.copy()

# Extract 'Base Attack' and 'Base Defense' attributes and Pokémon names
X = df_subset[['Base Attack', 'Base Defense']].values
names = df_subset['Name'].values

# Compute the distance matrix (Euclidean)
distance_matrix = pairwise_distances(X, metric='euclidean')

# Set the number of clusters
k = 2

# PAM algorithm initialization: randomly choose k medoids
np.random.seed(42)
n = X.shape[0]
medoid_indices = np.random.choice(n, k, replace=False)
current_cost = np.sum(np.min(distance_matrix[:, medoid_indices], axis=1))

print("########## K-MEDOIDS ##########\n")
print("Initial Medoids:")
for med in medoid_indices:
    print(f"  Index {med}: {names[med]}")
print(f"Initial cost: {current_cost}\n")

max_iter = 100  # Maximum number of outer iterations allowed
colors = ['red', 'blue']

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
                    color=colors[i], label=f'Cluster {i+1}')
    medoid_points = X[medoid_indices]
    plt.scatter(medoid_points[:, 0], medoid_points[:, 1],
                color='black', marker='X', s=200, label='Medoids')
    plt.title(f"K-Medoids Clustering - After Iteration {iteration}")
    plt.xlabel("Base Attack")
    plt.ylabel("Base Defense")
    plt.legend()
    plt.show()

# Final cluster assignment after convergence
labels = np.argmin(distance_matrix[:, medoid_indices], axis=1)
df_subset['KMedoids Cluster No'] = labels

# Plot final clustering result
plt.figure(figsize=(8, 6))
for i in range(k):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                color=colors[i], label=f'Cluster {i+1}')
plt.scatter(X[medoid_indices, 0], X[medoid_indices, 1],
            color='black', marker='X', s=200, label='Medoids')
plt.title("Final K-Medoids Clustering on Pokémon")
plt.xlabel("Base Attack")
plt.ylabel("Base Defense")
plt.legend()
plt.show()

# Display final cluster assignments
print("Final K-Medoids Cluster Assignments:")
for cluster in range(k):
    members = [names[i] for i in range(len(names)) if labels[i] == cluster]
    print(f"  Cluster {cluster+1}: {members}")
