print("""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

# Load your dataset
# Replace "path/to/your/dataset.csv" with the actual path to your dataset
data = pd.read_csv("path/to/your/dataset.csv")

# Select appropriate features and labels
X = data[['Feature1', 'Feature2', 'Feature3']]  # Adjust feature columns accordingly
y = data['Target']  # Adjust target column accordingly

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print decision tree rules
tree_rules = export_text(clf, feature_names=X.columns)
print("Decision Tree Rules:\n", tree_rules)
""")
