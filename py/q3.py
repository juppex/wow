code = """
import pandas as pd
import numpy as np

# Create a sample market basket dataset
data = {
 'transaction_id': [1,2,3,4,5,6],
 'product': ['Milk', 'Egg', 'Apples', 'Curd', 'Bread', 'Bananas'],
 'price': [23, 6, 40, 20, 50, 40],
 'category': ['Dairy','Farm','Grocery','Dairy','Bakery','Grocery']
}
df = pd.DataFrame(data)
bins=[0, 20, 35, np.inf]
labels=['Low', 'Medium', 'High']
# Discretize continuous variables (price)
df['price_category'] = pd.cut(df['price'], bins=bins, labels=labels)
# Display the DataFrame with the new columns
print("\\nDataFrame with discretized continuous variable (price):")
print(df)
"""

print(code)
