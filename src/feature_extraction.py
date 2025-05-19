import re

def urgency_score(email):
    urgency_words = {"urgent", "immediately", "limited", "winner", "offer", "now", "verify", "alert", "click", "act now", "asap", "important", "deadline", "risk", "warning"}
    return sum(1 for word in urgency_words if word in email.lower()) / len(urgency_words)

def politeness_score(email):
    politeness_words = {"please", "thank you", "kindly", "appreciate", "regards", "sincerely"}
    return sum(1 for word in politeness_words if word in email.lower()) / len(politeness_words)

def has_link(email):
    return 1 if re.search(r'https?://|www\.', email) else 0

def has_attachment(email):
    return 1 if 'attachment' in email.lower() or 'attached' in email.lower() else 0

def has_greeting(email):
    return 1 if re.search(r'\b(hi|hello|dear)\b', email.lower()) else 0

def has_farewell(email):
    return 1 if re.search(r'\b(thanks|regards|sincerely|best)\b', email.lower()) else 0

def excessive_caps(email):
    return 1 if sum(1 for c in email if c.isupper()) > len(email) * 0.3 else 0

def exclamation_density(email):
    return min(email.count('!') / len(email), 0.1)

def build_vocab(emails):
    vocab = set()
    for email in emails:
        for word in email.split():
            vocab.add(word.lower())
    return sorted(list(vocab))

def vectorize_email(email, vocab):
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    vector = [0] * len(vocab)
    for word in email.split():
        word = word.lower()
        if word in word_to_idx:
            vector[word_to_idx[word]] += 1

    # Custom Features
    vector.extend([
        urgency_score(email),
        politeness_score(email),
        has_link(email),
        has_attachment(email),
        has_greeting(email),
        has_farewell(email),
        excessive_caps(email),
        exclamation_density(email)
    ])
    return vector

def vectorize_emails(emails, vocab):
    return [vectorize_email(email, vocab) for email in emails]
