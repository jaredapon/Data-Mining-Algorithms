# Market Basket Analysis using Apriori Algorithm

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_rows', 500)  # Show all rows
pd.set_option('display.max_columns', 500)  # Show all columns
pd.set_option('display.width', 500)  # Auto-detect width
pd.set_option('display.max_colwidth', 500)  # Show full content in each cell
df = pd.read_csv('bakery_transactions.csv', low_memory=True)
    
# Convert the transactions to a one-hot encoded format
one_hot = df['Items'].str.get_dummies(sep=',')
    
# Ensure the DataFrame contains boolean values
one_hot = one_hot.astype(bool)

# Transactions
print("\n=== Jared's JaBread ===")
print("My Transaction Dataset\n")
transactions_df = pd.DataFrame({
    'Transaction_ID': df.iloc[:, 0],
    'Items': df['Items']
})
print(transactions_df)

# Generate frequent itemsets with a minimum support threshold
frequent_itemsets = apriori(one_hot, min_support=0.2, use_colnames=True)
print("\n=== Frequent Itemsets ===")
print(frequent_itemsets)

# Generate association rules from the frequent itemsets using a minimum confidence threshold
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.85)
rules = rules[
    (rules['lift'] >= 2.02) & 
    (rules['leverage'] > 0.01) &
    (rules['conviction'] > 1.2)
].sort_values(['lift', 'confidence'], ascending=[False, False])
    
print("\n=== Association Rules ===")
# Display the rule along with its support, confidence, and lift
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage', 'conviction']])