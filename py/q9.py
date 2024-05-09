code = """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

np.random.seed(0)
x = np.random.rand(100, 2)
y = (x[:, 0] + x[:, 1] > 1).astype(int)

X = x 
y = y 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

tree_rules = export_text(clf, feature_names=["Feature 1", "Feature 2"])
print("Decision Tree Rules:\\n", tree_rules)
"""

print(code)
