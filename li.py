print("""import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the Iris dataset
iris_data = pd.read_csv("iris.csv")

# Selecting appropriate features and labels
X = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_data['species']  # You might need to encode the target labels if using Linear Regression

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# For regression tasks, calculate evaluation metrics might not be applicable for classification
# But let's calculate the metrics to see
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Scatter plot might not be suitable for classification tasks

# Print evaluation metrics using f-strings with 2 decimal places
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Since this is a classification problem, you might want to evaluate using classification metrics such as accuracy, confusion matrix, etc.
# But here's how you would plot a scatter plot for the original data
# plt.scatter(X_test, y_test, color='red', label='Original Data')
# plt.plot(X_test, y_pred, color='orange', linewidth=3)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Linear Regression')
# plt.show()
""")
