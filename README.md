

# Email Spam Detection using Naive Bayes

A simple and interpretable spam detection system built using Python and the Naive Bayes algorithm. This project classifies email (or SMS) messages as **Spam** or **Ham (Not Spam)** using a Bag-of-Words model.

---

## Features

- Preprocessing of raw text (lowercasing, character cleaning, tokenization)
- Vocabulary building from training data
- Feature extraction via Bag-of-Words
- Naive Bayes classification
- Train/test split and accuracy evaluation
- Interactive prediction via UI
- Optional confusion matrix visualization

---

## Dataset

This project supports any labeled dataset in CSV format with two columns:


- `text`: the email or message content
- `label`: either `spam` or `ham`

---

##  How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/email-spam-detector.git
cd email-spam-detector

pip install -r requirements.txt
python main.py

.
├── main.py                 # Main script to run the spam detector
├── data/
│   └── spam.csv            # CSV file containing labeled messages
├── src/
│   ├── preprocessing.py    # Text cleaning and normalization
│   ├── feature_extraction.py # Vocabulary building and vectorization
│   ├── model.py            # Naive Bayes classifier implementation
│   ├── train.py            # Train/test splitting logic
│   ├── evaluate.py         # Accuracy computation
│   └── plot_confusion.py   # (Optional) Confusion matrix plotting


Enter an email to classify (or type 'exit' to quit):
> Congratulations! You've won a free iPhone! Click here to claim.
Prediction: spam

> Hey, are we still on for lunch today?
Prediction: ham





