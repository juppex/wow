code = """
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(42) # For reproducibility
x = 2*np.random.rand(100, 1) # Generate 100 random numbers between 0 and 1
y = 4+3*x+ np.random.randn(100,1) # Generate y as a linear function of X with some noise

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(x_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(x_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics using f-strings with 2 decimal places
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Scatter plot of the original data
plt.scatter(x_test, y_test, color='red', label='Original Data')
plt.plot(x_test,y_pred,color='orange',linewidth=3)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.show()

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
"""

print(code)
