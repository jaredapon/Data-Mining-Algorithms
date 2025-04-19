import pandas as pd 
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Sample dataset: 
transactions = [
   ['Phone Case', 'Screen Protector', 'Power Bank'],
   ['Phone Case', 'Wireless Earbuds'],
   ['Phone Case', 'Screen Protector'],
   ['Screen Protector', 'Wireless Earbuds'], 
   ['Phone Case', 'Screen Protector', 'Tripod'], 
   ['Phone Case', 'Power Bank'],
   ['Screen Protector', 'Wireless Earbuds'], 
   ['Phone Case', 'Screen Protector', 'Power Bank']
]


# Convert the list of transactions into a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

print("=== One-Hot Encoded Transaction Data ===")
print(df)

# Generate frequent itemsets with a minimum support threshold (e.g., 10%)
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
print("\n=== Frequent Itemsets ===")
print(frequent_itemsets)

# Generate association rules from the frequent itemsets using a minimum confidence threshold (e.g., 10%)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.1)
rules = rules[rules['conviction'] >= 1]

print("\n=== Association Rules ===")
# Display the rule along with its support, confidence, and lift
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage', 'conviction']])








