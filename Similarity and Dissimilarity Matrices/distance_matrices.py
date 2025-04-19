# This script calculates various distance matrices for a Pokémon dataset
# Hamming distance for nominal attributes
# Manhattan distance for numerical and ordinal attributes
# Symmetric Binary distance for binary attributes
# The final output is a Gower's Distance Matrix, an overall distance matrix that combines all these distances

import numpy as np
import pandas as pd

# Pokémon dataset
pokemon_data = pd.DataFrame({
    "Name": ["Duskull", "Cherrim", "Trubbish", "Cherubi", "Hakamo-o", "Primeape"],
    "Type": ["Ghost", "Grass", "Poison", "Grass", "Dragon/Fighting", "Fighting"],  # Nominal
    "Total Stats": [295, 450, 329, 275, 420, 455],  # Numerical
    "Evolution Stage": [1, 2, 1, 1, 2, 2],  # Ordinal
    "Legendary": [0, 0, 0, 0, 0, 0]  # Binary
})

print(pokemon_data, "\n")

# Hamming Distance for Types
unique_types = {t: i for i, t in enumerate(pokemon_data["Type"].unique())}
pokemon_data["Type Code"] = pokemon_data["Type"].map(unique_types)

type_codes = pokemon_data["Type Code"].values.reshape(-1, 1)
type_dist_matrix = np.not_equal(type_codes, type_codes.T).astype(float)

# Create DataFrame for Hamming Distance
type_dist_df = pd.DataFrame(type_dist_matrix, index=pokemon_data["Name"], columns=pokemon_data["Name"])
type_dist_df.index.name = None
type_dist_df.columns.name = None

print("Nominal Distance Matrix (Hamming Distance for Types)")
print(type_dist_df, "\n")

# Min-Max Scaling function
def min_max_scale(column, ordinal=False):
    min_val, max_val = column.min(), column.max()
    if ordinal:
        return (column - min_val) / (3 - 1)  # For ordinal data (m-1 scaling)
    return (column - min_val) / (max_val - min_val)  # Regular Min-Max Scaling

# Normalization
pokemon_data["Total Stats Scaled"] = min_max_scale(pokemon_data["Total Stats"])
pokemon_data["Evolution Stage Scaled"] = min_max_scale(pokemon_data["Evolution Stage"], ordinal=True)

# Manhattan Distance for Scaled Numerical
num_data = pokemon_data[["Total Stats Scaled"]].values.squeeze()  # Remove extra dimension
num_dist_matrix = np.abs(num_data[:, np.newaxis] - num_data[np.newaxis, :])

# Create DataFrame for Numerical Distance
num_dist_df = pd.DataFrame(num_dist_matrix, index=pokemon_data["Name"], columns=pokemon_data["Name"])
num_dist_df.index.name = None  # Remove the index name
num_dist_df.columns.name = None  # Remove the column name

print("Numerical Distance Matrix (Manhattan Distance for Stats)")
print(num_dist_df)

# Manhattan Distance for Scaled Ordinal
ord_data = pokemon_data[["Evolution Stage Scaled"]].values.squeeze()  # Remove extra dimension
ord_dist_matrix = np.abs(ord_data[:, np.newaxis] - ord_data[np.newaxis, :])

# Create DataFrame for Ordinal Distance
ord_dist_df = pd.DataFrame(ord_dist_matrix, index=pokemon_data["Name"], columns=pokemon_data["Name"])
ord_dist_df.index.name = None  # Remove the index name
ord_dist_df.columns.name = None  # Remove the column name

print("\nOrdinal Distance Matrix (Manhattan Distance for Evolution Stage)")
print(ord_dist_df)

# Binary Distance Matrix
legendary_status = pokemon_data["Legendary"].values
binary_dist_matrix = np.abs(legendary_status[:, np.newaxis] - legendary_status[np.newaxis, :]).astype(float)

# Create DataFrame for Binary Distance
binary_dist_df = pd.DataFrame(binary_dist_matrix, index=pokemon_data["Name"], columns=pokemon_data["Name"])
binary_dist_df.index.name = None  # Remove the index name
binary_dist_df.columns.name = None  # Remove the column name

print("\nBinary Distance Matrix (Symmetry for Legendary Status)")
print(binary_dist_df)

# Overall Distance Matrix (Gower's Dissimilarity)
distance_matrices = [type_dist_matrix, num_dist_matrix, ord_dist_matrix, binary_dist_matrix]
num_attributes = len(distance_matrices)  # Number of distance matrices
overall_distance_matrix = sum(distance_matrices) / num_attributes

# Create DataFrame for Overall Distance
overall_dist_df = pd.DataFrame(overall_distance_matrix, index=pokemon_data["Name"], columns=pokemon_data["Name"])
overall_dist_df.index.name = None  # Remove the index name
overall_dist_df.columns.name = None  # Remove the column name 
print("\nOverall Distance Matrix")
print(overall_dist_df)