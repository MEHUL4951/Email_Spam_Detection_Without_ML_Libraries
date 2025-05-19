# main.py

import csv
from src.preprocessing import preprocess_emails
from src.feature_extraction import build_vocab, vectorize_emails
from src.model import NaiveBayesClassifier
from src.train import train_test_split  # Your custom split with stratify
from src.evaluate import accuracy
from src.plot_confusion import plot_conf_matrix
from collections import Counter
from sklearn.metrics import classification_report

#Load Data
emails = []
labels = []

with open('data/spam_ham.csv', mode='r', encoding='utf8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    for row in reader:
        emails.append(row[0])
        labels.append(row[1])

print("Total emails:", len(emails))
print("Unique emails:", len(set(emails)))

# Deduplicate emails while keeping their label
seen = set()
unique_emails = []
unique_labels = []

for email, label in zip(emails, labels):
    if email not in seen:
        seen.add(email)
        unique_emails.append(email)
        unique_labels.append(label)

# Train/test split with stratification
x_train_raw, x_test_raw, y_train, y_test = train_test_split(
    unique_emails, unique_labels, test_size=0.2, stratify=unique_labels
)

# Preprocess
cleaned_train = preprocess_emails(x_train_raw)
cleaned_test = preprocess_emails(x_test_raw)

# Build vocab only from training data
vocab = build_vocab(cleaned_train)

# Vectorize
x_train = vectorize_emails(cleaned_train, vocab)
x_test = vectorize_emails(cleaned_test, vocab)

# Train
model = NaiveBayesClassifier(context_features_count=8)
model.train(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Report
print(classification_report(y_test, y_pred))
acc = accuracy(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")

# Check class balance
print("Train label distribution:", Counter(y_train))
print("Test label distribution:", Counter(y_test))

# Email classification loop
while True:
    user_input = input("\nEnter an email to classify (or type 'exit' to quit):\n> ")
    
    if user_input.lower() == 'exit':
        break
    
    cleaned = preprocess_emails([user_input])
    vector = vectorize_emails(cleaned, vocab)
    prediction = model.predict(vector)
    
    print(f"\nPrediction: {prediction[0]} (Spam/Ham)")
