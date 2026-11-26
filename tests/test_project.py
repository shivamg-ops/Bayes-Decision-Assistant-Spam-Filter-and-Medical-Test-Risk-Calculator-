"""
Unit Test Suite for Bayes Decision Assistant.
Covers:
- Medical probability calculations.
- Invalid percentage inputs.
- Zero denominator cases.
- Natural frequency calculations.
- Monte Carlo simulation.
- Spam prediction.
- Empty messages.
- Unknown words.
"""

import unittest
import numpy as np
import pandas as pd
from medical_calculator import calculate_bayes_medical
from natural_frequency import generate_natural_frequency_table
from monte_carlo import run_monte_carlo_simulation
from text_preprocessing import clean_text, tokenize
from spam_detection import NaiveBayesClassifier
from model_evaluation import compute_confusion_matrix, calculate_metrics


class TestBayesDecisionAssistant(unittest.TestCase):

    def setUp(self):
        # Sample dummy dataset for quick classifier testing
        self.sample_messages = [
            "Win cash prize claim now",
            "Urgent call free lottery award",
            "Hey are you coming to class today",
            "Let's meet at the library tomorrow",
            "Congratulations you won a free car call now",
            "Can you send me the lecture notes please"
        ]
        self.sample_labels = [
            "spam",
            "spam",
            "ham",
            "ham",
            "spam",
            "ham"
        ]
        self.model = NaiveBayesClassifier(alpha=1.0)
        self.model.fit(self.sample_messages, self.sample_labels)

    # 1. Medical Probability Calculations
    def test_medical_probability_calculation(self):
        # Known classic benchmark: Sensitivity=99%, Specificity=95%, Prevalence=0.1%
        # P(D) = 0.001, P(~D) = 0.999
        # P(+|D) = 0.99, P(+|~D) = 0.05
        # P(+) = (0.99 * 0.001) + (0.05 * 0.999) = 0.00099 + 0.04995 = 0.05094
        # P(D|+) = 0.00099 / 0.05094 ≈ 0.0194346 (1.94%)
        res = calculate_bayes_medical(sensitivity=99.0, specificity=95.0, prevalence=0.1)
        self.assertAlmostEqual(res['ppv_percentage'], 1.94346, places=3)
        self.assertGreater(res['npv_percentage'], 99.9)

    # 2. Invalid Percentage Inputs
    def test_invalid_percentage_inputs(self):
        with self.assertRaises(ValueError):
            calculate_bayes_medical(sensitivity=-5.0, specificity=90.0, prevalence=1.0)
        with self.assertRaises(ValueError):
            calculate_bayes_medical(sensitivity=95.0, specificity=150.0, prevalence=1.0)
        with self.assertRaises(ValueError):
            calculate_bayes_medical(sensitivity=95.0, specificity=90.0, prevalence=-1.0)

    # 3. Zero Denominator Cases
    def test_zero_denominator_case(self):
        # Sensitivity = 0 and Specificity = 100% (so P(+|~D) = 0)
        # Means nobody ever tests positive: P(+) = 0
        with self.assertRaises(ZeroDivisionError):
            calculate_bayes_medical(sensitivity=0.0, specificity=100.0, prevalence=5.0)

    # 4. Natural Frequency Calculations
    def test_natural_frequency_calculations(self):
        data = generate_natural_frequency_table(sensitivity=90.0, specificity=90.0, prevalence=1.0, population=1000)
        self.assertEqual(data['population'], 1000)
        self.assertAlmostEqual(data['num_with_disease'], 10.0)
        self.assertAlmostEqual(data['num_without_disease'], 990.0)
        self.assertAlmostEqual(data['true_positives'], 9.0)
        self.assertAlmostEqual(data['false_positives'], 99.0)
        self.assertAlmostEqual(data['total_positive_tests'], 108.0)
        self.assertAlmostEqual(data['calculated_percentage'], (9.0 / 108.0) * 100.0, places=4)

    # 5. Monte Carlo Simulation
    def test_monte_carlo_simulation(self):
        sim = run_monte_carlo_simulation(sensitivity=95.0, specificity=90.0, prevalence=2.0, num_samples=50000, random_seed=42)
        self.assertEqual(sim['num_samples'], 50000)
        # Simulation should be within 1.0% of theoretical Bayes
        self.assertLess(sim['difference_pct'], 1.0)
        self.assertEqual(sim['true_positives'] + sim['false_positives'], sim['total_positive_tests'])

    # 6. Text Preprocessing
    def test_text_preprocessing(self):
        raw = "Check this URL: http://spam.com, call 1800-123-456! FREE $$"
        cleaned = clean_text(raw)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("1800", cleaned)
        tokens = tokenize(raw)
        self.assertIn("free", tokens)
        self.assertIn("call", tokens)

    # 7. Spam Prediction
    def test_spam_prediction(self):
        pred_spam = self.model.predict("Claim your free cash prize now")
        self.assertEqual(pred_spam, "spam")
        pred_ham = self.model.predict("See you in the library for study")
        self.assertEqual(pred_ham, "ham")

    # 8. Empty Messages
    def test_empty_message(self):
        # Empty string should not crash; predicts based purely on prior
        empty_pred = self.model.predict("")
        probs = self.model.predict_proba("")
        self.assertIn(empty_pred, ["ham", "spam"])
        self.assertAlmostEqual(probs['ham'] + probs['spam'], 1.0, places=5)

    # 9. Unknown Words (Out-of-Vocabulary words handled by Laplace smoothing)
    def test_unknown_words(self):
        unknown_msg = "Zylophone qwertyuiop asdfghjkl"
        pred = self.model.predict(unknown_msg)
        probs = self.model.predict_proba(unknown_msg)
        self.assertIn(pred, ["ham", "spam"])
        self.assertAlmostEqual(probs['ham'] + probs['spam'], 1.0, places=5)

    # 10. Model Evaluation Metrics
    def test_metrics_calculation(self):
        y_true = ['spam', 'ham', 'spam', 'ham']
        y_pred = ['spam', 'spam', 'spam', 'ham']
        cm = compute_confusion_matrix(y_true, y_pred, pos_label='spam')
        self.assertEqual(cm['TP'], 2)
        self.assertEqual(cm['FP'], 1)
        self.assertEqual(cm['TN'], 1)
        self.assertEqual(cm['FN'], 0)
        metrics = calculate_metrics(y_true, y_pred, pos_label='spam')
        self.assertEqual(metrics['Recall'], 1.0)
        self.assertAlmostEqual(metrics['Precision'], 2/3, places=4)
        self.assertAlmostEqual(metrics['Accuracy'], 3/4, places=4)


if __name__ == '__main__':
    unittest.main()

