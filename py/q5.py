code = """
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample email data
emails = [
 {"text": "Get a free laptop now!", "label": "spam"},
 {"text": "Hey, how are you doing?", "label": "non-spam"}
 
]
# Construct text and labels data
texts = [email['text'] for email in emails]
labels = [email['label'] for email in emails]
# Split data into training and testing sets
texts_train, texts_test, labels_train, labels_test = train_test_split(texts,labels, test_size=0.2, random_state=42)
# Vectorize the text data
vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(texts_train)
x_test = vectorizer.transform(texts_test)
# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(x_train, labels_train)
predictions = classifier.predict(x_test)
accuracy = accuracy_score(labels_test, predictions)
classification_report_output = (classification_report(labels_test,predictions))
# Print accuracies
print(f"Accuracy: {accuracy:.2f}")
# Print classification report
print("\\nClassification Report:")
print(classification_report_output)
"""

print(code)
