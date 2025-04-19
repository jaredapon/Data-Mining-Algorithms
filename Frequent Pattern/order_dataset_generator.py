# This script generates a CSV file with 1000 transactions of bakery items.
# The dataset generated will be used for market basket analysis.

import csv
import random

breads = [
    "Pandesal", "Ensaymada", "Spanish Bread", "Monay", "Garlic Bread",
    "Banana Bread", "Pan de Coco", "Napoleones", "Pianono", "Mamon",
    "Taisan", "Hopia", "Senorita Bread", "Egg Pie", "Cheese Roll",
    "Ube Cheese Pandesal", "Bicho-Bicho", "Shakoy", "Pinagong", "Kababayan",
    "Lumpiang Saging", "Torta Cebuana", "Inipit", "Banana Bread", "Garlic Toast",
    "Jacobs", "Otap", "Rosquillos", "Barquillos", "Camachile"
]

transactions = []
for _ in range(1000):
    num_items = random.randint(1, 30)
    items = random.sample(breads, num_items)
    transactions.append(items)

with open("bakery_transactions.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Transaction_ID", "Items"])  #Header
    for i, items in enumerate(transactions, start=1):
        writer.writerow([i, ",".join(items)])

print("CSV file 'grocery_transactions.csv' with 1000 transactions has been generated")