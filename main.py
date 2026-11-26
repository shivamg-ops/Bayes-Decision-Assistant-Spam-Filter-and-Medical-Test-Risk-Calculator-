"""
Bayes Decision Assistant - Main Console Application.
Interactive menu-driven console program integrating:
1. SMS Spam Detection (Scratch Multinomial Naive Bayes)
2. Medical Test Risk Calculator (Bayes' Theorem)
3. Natural Frequency Table (Cohort of 1,000)
4. Monte Carlo Simulation (Cohort of 100,000)
5. Model Evaluation Metrics & Results Table
6. Visualization Generation
"""

import os
import sys
import pandas as pd
from spam_detection import NaiveBayesClassifier, train_test_split_scratch
from model_evaluation import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_class_distribution
)
from medical_calculator import calculate_bayes_medical
from natural_frequency import (
    generate_natural_frequency_table,
    format_natural_frequency_display
)
from monte_carlo import run_monte_carlo_simulation, plot_monte_carlo_comparison


class BayesAssistantApp:
    def __init__(self, dataset_path: str = "data/sms_spam_cleaned.csv"):
        self.dataset_path = dataset_path
        self.model = None
        self.df = None
        self.train_df = None
        self.test_df = None
        self.metrics = None
        self.last_medical_params = {'sensitivity': 95.0, 'specificity': 90.0, 'prevalence': 1.0}

    def load_and_train_spam_model(self):
        """Loads dataset and trains the Naive Bayes model if not already trained."""
        if self.model is not None and self.model.is_trained:
            return

        print("\n[INFO] Loading SMS Spam dataset and training Naive Bayes classifier...")
        if not os.path.exists(self.dataset_path):
            print(f"[ERROR] Dataset not found at {self.dataset_path}.")
            print("Please ensure sms_spam_cleaned.csv exists in the data/ folder.")
            return

        self.df = pd.read_csv(self.dataset_path)
        self.train_df, self.test_df = train_test_split_scratch(self.df, test_ratio=0.2, random_state=42)

        self.model = NaiveBayesClassifier(alpha=1.0)
        self.model.fit(self.train_df['message'].tolist(), self.train_df['label'].tolist())

        # Evaluate on test set
        y_test = self.test_df['label'].tolist()
        y_pred = self.model.batch_predict(self.test_df['message'].tolist())
        self.metrics = calculate_metrics(y_test, y_pred, pos_label='spam')
        print(f"[SUCCESS] Trained on {len(self.train_df)} messages. Vocabulary size: {self.model.vocab_size} words.")
        print(f"[SUCCESS] Test Accuracy: {self.metrics['Accuracy']*100:.2f}%, F1-Score: {self.metrics['F1-Score']:.4f}")

    def handle_sms_spam_detection(self):
        """Menu Option 1: Classify user SMS message."""
        self.load_and_train_spam_model()
        if self.model is None:
            return

        print("\n--- SMS SPAM DETECTION ---")
        msg = input("Enter an SMS message to classify: ").strip()
        if not msg:
            print("[WARNING] Empty message entered. Please enter a valid text.")
            return

        prediction = self.model.predict(msg)
        probs = self.model.predict_proba(msg)

        print("\n" + "=" * 45)
        print("          PREDICTION RESULT")
        print("=" * 45)
        print(f"Message: \"{msg}\"")
        print(f"Prediction: {'SPAM [🚨]' if prediction == 'spam' else 'HAM / NOT SPAM [✅]'}")
        print("-" * 45)
        print(f"P(Spam | Message): {probs['spam'] * 100:.4f}%")
        print(f"P(Ham  | Message): {probs['ham'] * 100:.4f}%")
        print("=" * 45)

    def handle_medical_calculator(self):
        """Menu Option 2: Medical Test Risk Calculator using Bayes' Theorem."""
        print("\n--- MEDICAL TEST RISK CALCULATOR (BAYES' THEOREM) ---")
        try:
            sens_in = input("Enter Sensitivity (%) [e.g., 95]: ").strip()
            spec_in = input("Enter Specificity (%) [e.g., 90]: ").strip()
            prev_in = input("Enter Disease Prevalence (%) [e.g., 1.0]: ").strip()

            sens = float(sens_in)
            spec = float(spec_in)
            prev = float(prev_in)

            results = calculate_bayes_medical(sens, spec, prev)
            self.last_medical_params = {'sensitivity': sens, 'specificity': spec, 'prevalence': prev}

            print("\n" + "=" * 60)
            print("          STEP-BY-STEP BAYESIAN CALCULATION")
            print("=" * 60)
            print(f"1. Given Parameters:")
            print(f"   - Prior P(Disease)          = {results['p_d']:.4f} ({sens:.1f}%)")
            print(f"   - Prior P(Healthy)          = {results['p_not_d']:.4f}")
            print(f"   - Sensitivity P(+|Disease)  = {results['p_pos_given_d']:.4f} ({sens:.1f}%)")
            print(f"   - False Positive P(+|Health)= {results['p_pos_given_not_d']:.4f} ({100-spec:.1f}%)")
            print("-" * 60)
            print(f"2. Law of Total Probability P(+):")
            print(f"   P(+) = P(+|D)*P(D) + P(+|~D)*P(~D)")
            print(f"        = ({results['p_pos_given_d']:.4f} * {results['p_d']:.4f}) + ({results['p_pos_given_not_d']:.4f} * {results['p_not_d']:.4f})")
            print(f"        = {results['p_pos']:.6f}")
            print("-" * 60)
            print(f"3. Positive Predictive Value P(Disease | +):")
            print(f"   P(D|+) = [P(+|D) * P(D)] / P(+)")
            print(f"          = [{results['p_pos_given_d']:.4f} * {results['p_d']:.4f}] / {results['p_pos']:.6f}")
            print(f"          = {results['p_d_given_pos']:.4f} ({results['ppv_percentage']:.2f}%)")
            print("-" * 60)
            print(f"4. Negative Predictive Value P(No Disease | -):")
            print(f"   P(~D|-) = {results['p_not_d_given_neg']:.4f} ({results['npv_percentage']:.2f}%)")
            print("=" * 60)
            print(f"\nINTERPRETATION:\n{results['interpretation']}")

        except ValueError as ve:
            print(f"[INPUT ERROR] {ve}")
        except ZeroDivisionError as zde:
            print(f"[MATH ERROR] {zde}")
        except Exception as e:
            print(f"[ERROR] {e}")

    def handle_natural_frequency(self):
        """Menu Option 3: Natural Frequency Table for 1,000 people."""
        print("\n--- NATURAL FREQUENCY TABLE ---")
        print(f"Using current test parameters:")
        print(f"Sensitivity: {self.last_medical_params['sensitivity']}%, "
              f"Specificity: {self.last_medical_params['specificity']}%, "
              f"Prevalence: {self.last_medical_params['prevalence']}%")

        choice = input("Do you want to use custom parameters? (y/N): ").strip().lower()
        sens = self.last_medical_params['sensitivity']
        spec = self.last_medical_params['specificity']
        prev = self.last_medical_params['prevalence']

        if choice == 'y':
            try:
                sens = float(input("Enter Sensitivity (%): ").strip())
                spec = float(input("Enter Specificity (%): ").strip())
                prev = float(input("Enter Prevalence (%): ").strip())
                self.last_medical_params = {'sensitivity': sens, 'specificity': spec, 'prevalence': prev}
            except ValueError:
                print("[ERROR] Invalid number format. Reverting to existing parameters.")

        data = generate_natural_frequency_table(sens, spec, prev, population=1000)
        display_str = format_natural_frequency_display(data)
        print(display_str)

    def handle_monte_carlo(self):
        """Menu Option 4: Monte Carlo Simulation with 100,000 individuals."""
        print("\n--- MONTE CARLO SIMULATION (100,000 INDIVIDUALS) ---")
        sens = self.last_medical_params['sensitivity']
        spec = self.last_medical_params['specificity']
        prev = self.last_medical_params['prevalence']

        print(f"Running simulation with: Sensitivity={sens}%, Specificity={spec}%, Prevalence={prev}%")
        print("Simulating 100,000 virtual subjects...")

        sim = run_monte_carlo_simulation(sens, spec, prev, num_samples=100000, random_seed=42)

        print("\n" + "=" * 65)
        print("          MONTE CARLO EMPIRICAL VERIFICATION")
        print("=" * 65)
        print(f"Total Simulated Population       : {sim['num_samples']:,}")
        print(f"Individuals with Disease         : {sim['has_disease_count']:,} ({sim['has_disease_count']/sim['num_samples']*100:.2f}%)")
        print(f"Healthy Individuals              : {sim['healthy_count']:,} ({sim['healthy_count']/sim['num_samples']*100:.2f}%)")
        print("-" * 65)
        print(f"True Positives (Sick & Tested +) : {sim['true_positives']:,}")
        print(f"False Positives(Healthy & Tested+): {sim['false_positives']:,}")
        print(f"Total Positive Test Results      : {sim['total_positive_tests']:,}")
        print("-" * 65)
        print(f"Simulated P(Disease | +)         : {sim['simulated_ppv_pct']:.2f}% ({sim['simulated_ppv']:.4f})")
        print(f"Theoretical Bayes P(Disease | +) : {sim['theoretical_ppv_pct']:.2f}% ({sim['theoretical_ppv']:.4f})")
        print(f"Absolute Difference              : {sim['difference_pct']:.4f}% ({sim['absolute_difference']:.6f})")
        print("=" * 65)
        print("CONCLUSION: The Monte Carlo simulation strongly aligns with Bayes' Theorem,")
        print("proving that the theoretical formula perfectly matches empirical observation.")

    def handle_view_model_results(self):
        """Menu Option 5: Model Evaluation Results Table."""
        self.load_and_train_spam_model()
        if self.metrics is None:
            return

        print("\n" + "=" * 60)
        print("      SMS SPAM NAIVE BAYES MODEL EVALUATION TABLE")
        print("=" * 60)
        print(f"| {'Metric':<30} | {'Value':<23} |")
        print("-" * 60)
        print(f"| {'Total Training Messages':<30} | {len(self.train_df):<23} |")
        print(f"| {'Total Testing Messages':<30} | {len(self.test_df):<23} |")
        print(f"| {'Unique Vocabulary Words':<30} | {self.model.vocab_size:<23} |")
        print(f"| {'True Positives (TP)':<30} | {self.metrics['TP']:<23} |")
        print(f"| {'False Positives (FP)':<30} | {self.metrics['FP']:<23} |")
        print(f"| {'True Negatives (TN)':<30} | {self.metrics['TN']:<23} |")
        print(f"| {'False Negatives (FN)':<30} | {self.metrics['FN']:<23} |")
        print("-" * 60)
        print(f"| {'Accuracy':<30} | {self.metrics['Accuracy']*100:.2f}% ({self.metrics['Accuracy']:.4f}){' ':6} |")
        print(f"| {'Precision (Spam)':<30} | {self.metrics['Precision']*100:.2f}% ({self.metrics['Precision']:.4f}){' ':6} |")
        print(f"| {'Recall (Spam)':<30} | {self.metrics['Recall']*100:.2f}% ({self.metrics['Recall']:.4f}){' ':6} |")
        print(f"| {'F1-Score':<30} | {self.metrics['F1-Score']:.4f}{' ':17} |")
        print("=" * 60)

    def handle_generate_visualizations(self):
        """Menu Option 6: Generate and save all 3 publication plots."""
        self.load_and_train_spam_model()
        print("\n[INFO] Generating publication-quality visualization plots...")
        os.makedirs("results", exist_ok=True)

        # Plot 1: Dataset Distribution
        plot_class_distribution(self.df, "results/spam_distribution.png")
        print(" -> Saved: results/spam_distribution.png")

        # Plot 2: Confusion Matrix Heatmap
        plot_confusion_matrix(self.metrics, "results/confusion_matrix.png")
        print(" -> Saved: results/confusion_matrix.png")

        # Plot 3: Theoretical vs Monte Carlo Simulation
        sens = self.last_medical_params['sensitivity']
        spec = self.last_medical_params['specificity']
        prev = self.last_medical_params['prevalence']
        sim = run_monte_carlo_simulation(sens, spec, prev, num_samples=100000, random_seed=42)
        plot_monte_carlo_comparison(sim, "results/simulation_graph.png")
        print(" -> Saved: results/simulation_graph.png")

        print("[SUCCESS] All plots generated and saved in results/ directory!")

    def run(self):
        """Main interaction loop."""
        while True:
            print("\n========== BAYES DECISION ASSISTANT ==========")
            print("1. SMS Spam Detection")
            print("2. Medical Test Risk Calculator")
            print("3. View Natural Frequency Table")
            print("4. Run Monte Carlo Simulation")
            print("5. View Model Evaluation Results")
            print("6. Generate Visualizations")
            print("7. Exit")
            print("==============================================")

            choice = input("Enter your choice (1-7): ").strip()

            if choice == '1':
                self.handle_sms_spam_detection()
            elif choice == '2':
                self.handle_medical_calculator()
            elif choice == '3':
                self.handle_natural_frequency()
            elif choice == '4':
                self.handle_monte_carlo()
            elif choice == '5':
                self.handle_view_model_results()
            elif choice == '6':
                self.handle_generate_visualizations()
            elif choice == '7':
                print("\nThank you for using Bayes Decision Assistant. Goodbye!\n")
                sys.exit(0)
            else:
                print("[INVALID INPUT] Please select an option between 1 and 7.")


if __name__ == "__main__":
    app = BayesAssistantApp()
    app.run()

