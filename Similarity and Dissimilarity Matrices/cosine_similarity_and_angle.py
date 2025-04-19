# Cosine Similarity and Angles Between Pokémon Stat Vectors
# It uses matplotlib to create a quiver plot for each pair of Pokémon.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data for six Pokémon
data = {
    'Pokemon': ['Duskull', 'Cherrim', 'Trubbish', 'Cherubi', 'Hakamo-o', 'Primeape'],
    'HP': [20, 70, 50, 45, 55, 65],
    'Attack': [40, 60, 50, 35, 75, 105],
    'Defense': [90, 70, 62, 45, 90, 60],
    'Sp. Atk': [30, 87, 40, 62, 65, 60],
    'Sp. Def': [90, 78, 62, 53, 70, 70],
    'Speed': [25, 85, 65, 35, 65, 95],
}

# Create DataFrame and set index
pokemon_df = pd.DataFrame(data)
pokemon_df.set_index('Pokemon', inplace=True)
pokemon_df.index.name = None

# Print Data Matrix
print("My Pokemon Data Matrix:")
print(pokemon_df.to_string(), "\n")

# Function to compute cosine similarity
def cosine_similarity_matrix(df):
    stats = df.values
    norm = np.linalg.norm(stats, axis=1)
    similarity_matrix = np.dot(stats, stats.T) / (norm[:, None] * norm[None, :])
    return pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)

# Calculate cosine similarity
cosine_sim_df = cosine_similarity_matrix(pokemon_df)

# Print Cosine Similarity Matrix (decimal form)
print("Cosine Similarity Matrix (Decimals):")
print(cosine_sim_df.round(2).to_string(), "\n")

# Convert cosine similarity to angles in degrees
angles_rad = np.arccos(np.clip(cosine_sim_df.values, -1.0, 1.0))
angles_deg = np.degrees(angles_rad)
angular_sim_df = pd.DataFrame(angles_deg, index=cosine_sim_df.index, columns=cosine_sim_df.columns)

# Print Cosine Similarity Matrix (degrees)
print("Cosine Similarity Matrix (Degrees):")
print(angular_sim_df.round(2).to_string())

# List Pokémon names and extract unique pairs (upper triangle)
pokemon_names = angular_sim_df.index.tolist()
pairs = [(i, j) for idx, i in enumerate(pokemon_names)
         for j in pokemon_names[idx+1:]]

n_pairs = len(pairs)
n_cols = 3  # Change to 3 columns
n_rows = (n_pairs + n_cols - 1) // n_cols  # Automatically calculate rows

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
axes = axes.flatten()

# Plot angles for each pair
for ax, (poke1, poke2) in zip(axes, pairs):
    angle_deg = angular_sim_df.loc[poke1, poke2]
    angle_rad = np.deg2rad(angle_deg)
    
    # Unit vectors: v1 is [1, 0]; v2 is rotated by angle_rad
    v1 = np.array([1, 0])
    v2 = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='red', label=poke1)
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='blue', label=poke2)
    
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    
    # Set the title for each subplot
    ax.set_title(f"{poke1} vs {poke2}\n(Angle: {angle_deg:.2f}°)")

# Remove empty subplots if any
for i in range(len(pairs), len(axes)):
    fig.delaxes(axes[i])

plt.suptitle("Cosine Similarity (Degrees) Between Pokémon Stat Vectors", fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()