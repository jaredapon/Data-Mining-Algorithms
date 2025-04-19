import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Set environment variable to avoid potential issues on Windows
os.environ["OMP_NUM_THREADS"] = '1'

# Load the Pokémon dataset
df = pd.read_csv(r"C:\Users\Jared Apon\Desktop\PokemonData.csv")
df_subset = df.copy()

# Extract 'Base Attack', 'Base Defense' attributes and Pokémon names
X = df_subset[['Base Attack', 'Base Defense']].values
names = df_subset['Name'].values

# Set the number of clusters
k = 3

# Initialize medians randomly from the data points
np.random.seed(42)
initial_indices = np.random.choice(len(X), k, replace=False)
medians = X[initial_indices]

print("########## K-MEDIANS ##########\n")

max_iters = 100
colors = ['red', 'blue', 'yellow']

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
