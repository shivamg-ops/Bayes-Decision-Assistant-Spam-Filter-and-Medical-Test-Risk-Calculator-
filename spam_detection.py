"""
Naive Bayes Classifier for SMS Spam Detection (Scratch Implementation).
Uses pure Python and NumPy to implement Multinomial Naive Bayes with Laplace Smoothing.
No scikit-learn dependency is used for classification.
"""

import math
from collections import defaultdict
import numpy as np
import pandas as pd
from text_preprocessing import tokenize


class NaiveBayesClassifier:
    """
    Multinomial Naive Bayes Classifier from scratch for binary text classification.
    Classes: 'spam' vs 'ham'.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initializes the classifier.

        Args:
            alpha (float): Laplace smoothing parameter (default = 1.0).
        """
        self.alpha = alpha
        self.classes = ['ham', 'spam']
        self.class_counts = {c: 0 for c in self.classes}
        self.class_priors = {c: 0.0 for c in self.classes}
        self.word_counts = {c: defaultdict(int) for c in self.classes}
        self.total_words_in_class = {c: 0 for c in self.classes}
        self.vocab = set()
        self.vocab_size = 0
        self.is_trained = False

    def fit(self, messages: list, labels: list):
        """
        Trains the classifier on labeled text messages.

        Args:
            messages (list of str): SMS message strings.
            labels (list of str): Target labels ('ham' or 'spam').
        """
        if len(messages) != len(labels):
            raise ValueError("Messages and labels must be of the same length.")

        total_samples = len(messages)
        if total_samples == 0:
            raise ValueError("Training set cannot be empty.")

        # Reset counts
        self.class_counts = {c: 0 for c in self.classes}
        self.word_counts = {c: defaultdict(int) for c in self.classes}
        self.total_words_in_class = {c: 0 for c in self.classes}
        self.vocab = set()

        # Step 1: Count documents and words per class
        for msg, label in zip(messages, labels):
            label = str(label).strip().lower()
            if label not in self.classes:
                continue

            self.class_counts[label] += 1
            tokens = tokenize(msg)
            for token in tokens:
                self.word_counts[label][token] += 1
                self.total_words_in_class[label] += 1
                self.vocab.add(token)

        self.vocab_size = len(self.vocab)

        # Step 2: Compute prior probabilities P(Class)
        for c in self.classes:
            self.class_priors[c] = self.class_counts[c] / total_samples

        self.is_trained = True
        return self

    def _calculate_log_likelihood(self, token: str, class_label: str) -> float:
        """
        Calculates log P(word | class) with Laplace smoothing:
        P(w|c) = (count(w, c) + alpha) / (total_words(c) + alpha * |V|)
        """
        count = self.word_counts[class_label].get(token, 0)
        numerator = count + self.alpha
        denominator = self.total_words_in_class[class_label] + (self.alpha * self.vocab_size)
        return math.log(numerator / denominator)

    def predict_log_proba(self, message: str) -> dict:
        """
        Computes log posterior score for each class given a message:
        log P(c | d) = log P(c) + sum(log P(w_i | c))
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting.")

        tokens = tokenize(message)
        log_posteriors = {}

        for c in self.classes:
            # Prior log probability
            log_prob = math.log(self.class_priors[c]) if self.class_priors[c] > 0 else -1e9
            # Sum of word log likelihoods
            for token in tokens:
                log_prob += self._calculate_log_likelihood(token, c)
            log_posteriors[c] = log_prob

        return log_posteriors

    def predict_proba(self, message: str) -> dict:
        """
        Computes normalized posterior probabilities using softmax over log scores.
        Prevents numerical underflow.
        """
        log_scores = self.predict_log_proba(message)
        max_log = max(log_scores.values())

        # Exponentiate shifted log scores to avoid overflow/underflow
        exp_scores = {c: math.exp(score - max_log) for c, score in log_scores.items()}
        total_exp = sum(exp_scores.values())

        probabilities = {c: exp_scores[c] / total_exp for c in self.classes}
        return probabilities

    def predict(self, message: str) -> str:
        """
        Predicts class ('ham' or 'spam') with maximum posterior probability.
        """
        probabilities = self.predict_proba(message)
        return 'spam' if probabilities['spam'] >= probabilities['ham'] else 'ham'

    def batch_predict(self, messages: list) -> list:
        """
        Predicts labels for a list of messages.
        """
        return [self.predict(msg) for msg in messages]


def train_test_split_scratch(df: pd.DataFrame, test_ratio: float = 0.2, random_state: int = 42):
    """
    Splits DataFrame into train and test sets from scratch using NumPy permutation.
    """
    np.random.seed(random_state)
    indices = np.random.permutation(len(df))
    test_size = int(len(df) * test_ratio)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    return train_df, test_df

