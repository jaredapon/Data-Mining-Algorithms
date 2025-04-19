# This script computes the Manhattan, Euclidean, and supremum distance matrices for a set of Pokémon based on their attributes.

import numpy as np
import pandas as pd

# Define the Pokémon data
data = {
    "Duskull": [20, 40, 90, 30, 90, 25],
    "Cherrim": [70, 60, 70, 87, 78, 85],
    "Trubbish": [50, 50, 62, 40, 62, 65],
    "Cherubi": [45, 35, 45, 62, 53, 35],
    "Hakamo-o": [55, 75, 90, 65, 70, 65],
    "Primeape": [65, 105, 60, 60, 70, 95]
}

column_names = ["HP", "Attack", "Defense", "Sp Attack", "Sp. Def", "Speed"]

# Convert to numpy array
names = list(data.keys())
attribute_matrix = np.array(list(data.values()))

# Create DataFrame with custom column names
df_attributes = pd.DataFrame(attribute_matrix, columns=column_names, index=names)

# Distance functions
def manhattan_distance(A, B):
    return np.sum(np.abs(A - B))

def euclidean_distance(A, B):
    return np.sqrt(np.sum((A - B) ** 2))

def supremum_distance(A, B):
    return np.max(np.abs(A - B))

# Compute the distance matrices
num_objects = len(attribute_matrix)  # number of objects
manhattan_matrix = np.zeros((num_objects, num_objects))
euclidean_matrix = np.zeros((num_objects, num_objects))
supremum_matrix = np.zeros((num_objects, num_objects))

for i in range(num_objects):  # loop over rows (objects)
    for j in range(num_objects):  # loop over columns (objects)
        manhattan_matrix[i, j] = manhattan_distance(attribute_matrix[i], attribute_matrix[j])
        euclidean_matrix[i, j] = euclidean_distance(attribute_matrix[i], attribute_matrix[j])
        supremum_matrix[i, j] = supremum_distance(attribute_matrix[i], attribute_matrix[j])

df_manhattan = pd.DataFrame(manhattan_matrix, columns=names, index=names)
df_euclidean = pd.DataFrame(euclidean_matrix, columns=names, index=names)
df_supremum = pd.DataFrame(supremum_matrix, columns=names, index=names)

print("Numerical Attribute Matrix")
print(df_attributes)

print("\nManhattan Distance Matrix:")
print(df_manhattan)

print("\nEuclidean Distance Matrix:")
print(df_euclidean)

print("\nSupremum Distance Matrix")
print(df_supremum)