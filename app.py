# app.py
from flask import Flask, render_template, request, jsonify
from src.preprocessing import preprocess_emails
from src.feature_extraction import build_vocab, vectorize_emails
from src.model import NaiveBayesClassifier
import csv
from collections import defaultdict

app = Flask(__name__)

# Load and initialize model (similar to main.py)
def initialize_model():
    # Load Data
    emails = []
    labels = []

    with open('data/spam_ham.csv', mode='r', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            emails.append(row[0])
            labels.append(row[1])

    # Deduplicate emails while keeping their label
    seen = set()
    unique_emails = []
    unique_labels = []

    for email, label in zip(emails, labels):
        if email not in seen:
            seen.add(email)
            unique_emails.append(email)
            unique_labels.append(label)

    # Preprocess
    cleaned_emails = preprocess_emails(unique_emails)

    # Build vocab
    vocab = build_vocab(cleaned_emails)

    # Vectorize
    x = vectorize_emails(cleaned_emails, vocab)

    # Train model
    model = NaiveBayesClassifier(context_features_count=8)
    model.train(x, unique_labels)

    return model, vocab

model, vocab = initialize_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    email_text = request.form.get('email_text', '')
    
    if not email_text:
        return jsonify({'error': 'No email text provided'}), 400
    
    # Process the email
    cleaned = preprocess_emails([email_text])
    vector = vectorize_emails(cleaned, vocab)
    prediction, spam_prob, ham_prob = model.predict_one(vector[0])
    
    return jsonify({
        'prediction': prediction,
        'email_text': email_text,
        'spam_prob': round(spam_prob * 100, 2),
        'ham_prob': round(ham_prob * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)