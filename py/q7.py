code = """
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

# Generate random binary classification data
np.random.seed(42)
x = 2 * np.random.rand(100, 1)
y = (4 + 3 * x + np.random.randn(100, 1)) > 5

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(x_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate classification report
class_report = classification_report(y_test, y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print confusion matrix, classification report, and accuracy
print(f"Confusion Matrix:\\n{conf_matrix}\\n")
print(f"Classification Report:\\n{class_report}\\n")
print(f"Accuracy: {accuracy:.2f}")

# Plot scatter plot of original data
plt.scatter(x_test, y_test, color='red')

# Plot predicted probabilities
plt.plot(x_test,model.predict_proba(x_test) [:,1], color='green', linewidth=4)
plt.xlabel('X')
plt.ylabel('Probability')
plt.title('Logistic Regression')
plt.legend()
plt.show()
"""

print(code)
