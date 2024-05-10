print("""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset from a local file
iris_data = pd.read_csv("path/to/your/iris.csv")  # Replace "path/to/your/iris.csv" with the actual path

# Encoding the target labels into binary classes
iris_data['species'] = iris_data['species'].map({'setosa': 0, 'versicolor': 0, 'virginica': 1})

# Selecting appropriate features and labels
X = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_data['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate classification report
class_report = classification_report(y_test, y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print confusion matrix, classification report, and accuracy
print(f"Confusion Matrix:\n{conf_matrix}\n")
print(f"Classification Report:\n{class_report}\n")
print(f"Accuracy: {accuracy:.2f}")
""")
