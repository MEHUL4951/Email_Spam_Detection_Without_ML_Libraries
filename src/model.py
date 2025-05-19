import math

class NaiveBayesClassifier:
    def __init__(self, context_features_count=8):
        self.vocab = []
        self.spam_word_probs = {}
        self.ham_word_probs = {}
        self.spam_prior = 0
        self.ham_prior = 0
        self.vocab_size = 0
        self.context_features_count = context_features_count
        self.spam_feature_avgs = []
        self.ham_feature_avgs = []

    def train(self, x_train, y_train):
        spam_emails = [x_train[i] for i in range(len(y_train)) if y_train[i] == 'spam']
        ham_emails  = [x_train[i] for i in range(len(y_train)) if y_train[i] == 'ham']

        self.spam_prior = (len(spam_emails)+1) / (len(x_train)+2)
        self.ham_prior = (len(ham_emails)+1) / (len(x_train)+2)

        vocab_len = len(x_train[0]) - self.context_features_count
        spam_word_count = [sum(vec[i] for vec in spam_emails) for i in range(vocab_len)]
        ham_word_count  = [sum(vec[i] for vec in ham_emails) for i in range(vocab_len)]

        self.vocab_size = len(spam_word_count)
        total_spam_words = sum(spam_word_count)
        total_ham_words = sum(ham_word_count)

        self.spam_word_probs = [(count + 1) / (total_spam_words + self.vocab_size) for count in spam_word_count]
        self.ham_word_probs = [(count + 1) / (total_ham_words + self.vocab_size) for count in ham_word_count]

        def avg_feature_values(emails):
            sums = [0.0] * self.context_features_count
            for email in emails:
                features = email[-self.context_features_count:]
                for i in range(self.context_features_count):
                    sums[i] += features[i]
            return [s / len(emails) for s in sums]

        self.spam_feature_avgs = avg_feature_values(spam_emails)
        self.ham_feature_avgs = avg_feature_values(ham_emails)

    def predict_one(self, email_vector):
        vocab_part = email_vector[:-self.context_features_count]
        context_part = email_vector[-self.context_features_count:]

        spam_score = math.log(self.spam_prior)
        ham_score = math.log(self.ham_prior)

        for i, count in enumerate(vocab_part):
            if count > 0:
                spam_score += count * math.log(self.spam_word_probs[i])
                ham_score += count * math.log(self.ham_word_probs[i])

        for i in range(self.context_features_count):
            feature_value = context_part[i]
            spam_avg = self.spam_feature_avgs[i]
            ham_avg = self.ham_feature_avgs[i]

            spam_score += -abs(feature_value - spam_avg)
            ham_score += -abs(feature_value - ham_avg)

        # Convert to probabilities using softmax
        max_score = max(spam_score, ham_score)
        exp_spam = math.exp(spam_score - max_score)
        exp_ham = math.exp(ham_score - max_score)
        spam_prob = exp_spam / (exp_spam + exp_ham)
        ham_prob = exp_ham / (exp_spam + exp_ham)

        prediction = 'spam' if spam_prob > ham_prob else 'ham'
        return prediction, spam_prob, ham_prob
    def predict(self, X_test):
        return [self.predict_one(vec) for vec in X_test]
