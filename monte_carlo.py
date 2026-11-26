"""
Monte Carlo Simulation Module.
Simulates a population of 100,000 individuals using NumPy random sampling
to empirically verify Bayes' Theorem and Positive Predictive Value.
Generates comparison graphs and convergence traces.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from medical_calculator import calculate_bayes_medical


def run_monte_carlo_simulation(sensitivity: float, specificity: float, prevalence: float,
                               num_samples: int = 100000, random_seed: int = 42) -> dict:
    """
    Runs a Monte Carlo simulation of `num_samples` individuals.

    Args:
        sensitivity (float): Sensitivity percentage (0 to 100).
        specificity (float): Specificity percentage (0 to 100).
        prevalence (float): Prevalence percentage (0 to 100).
        num_samples (int): Number of individuals to simulate (default = 100,000).
        random_seed (int): Fixed seed for reproducibility.

    Returns:
        dict: Simulation counts, empirical probabilities, theoretical values, and difference.
    """
    np.random.seed(random_seed)

    sens = sensitivity / 100.0
    spec = specificity / 100.0
    prev = prevalence / 100.0

    # Step 1: Assign disease status based on prevalence
    # 1 = Disease Present, 0 = Disease Absent
    has_disease = (np.random.rand(num_samples) < prev).astype(int)

    # Step 2: Simulate test results based on sensitivity and specificity
    # For sick people (has_disease == 1), test is positive with probability = sensitivity
    # For healthy people (has_disease == 0), test is positive with probability = (1 - specificity)
    rand_tests = np.random.rand(num_samples)
    test_positive = np.zeros(num_samples, dtype=int)

    test_positive[has_disease == 1] = (rand_tests[has_disease == 1] < sens).astype(int)
    test_positive[has_disease == 0] = (rand_tests[has_disease == 0] < (1.0 - spec)).astype(int)

    # Step 3: Count empirical occurrences
    true_positives = int(np.sum((has_disease == 1) & (test_positive == 1)))
    false_positives = int(np.sum((has_disease == 0) & (test_positive == 1)))
    true_negatives = int(np.sum((has_disease == 0) & (test_positive == 0)))
    false_negatives = int(np.sum((has_disease == 1) & (test_positive == 0)))

    total_positive_tests = true_positives + false_positives
    simulated_ppv = (true_positives / total_positive_tests) if total_positive_tests > 0 else 0.0

    # Step 4: Theoretical Bayes computation
    theoretical_calc = calculate_bayes_medical(sensitivity, specificity, prevalence)
    theoretical_ppv = theoretical_calc['p_d_given_pos']

    diff = abs(simulated_ppv - theoretical_ppv)
    diff_pct = abs((simulated_ppv * 100.0) - theoretical_calc['ppv_percentage'])

    return {
        'num_samples': num_samples,
        'has_disease_count': int(np.sum(has_disease)),
        'healthy_count': int(np.sum(has_disease == 0)),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'total_positive_tests': total_positive_tests,
        'simulated_ppv': simulated_ppv,
        'simulated_ppv_pct': simulated_ppv * 100.0,
        'theoretical_ppv': theoretical_ppv,
        'theoretical_ppv_pct': theoretical_calc['ppv_percentage'],
        'absolute_difference': diff,
        'difference_pct': diff_pct
    }


def plot_monte_carlo_comparison(sim_results: dict, save_path: str = 'results/simulation_graph.png'):
    """
    Plots a side-by-side bar chart and convergence chart comparing Theoretical vs Simulated values.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Bar Chart Comparison
    categories = ['Theoretical Bayes', 'Monte Carlo Sim']
    values = [sim_results['theoretical_ppv_pct'], sim_results['simulated_ppv_pct']]
    colors = ['#2b5c8f', '#e26d5c']

    bars = ax1.bar(categories, values, color=colors, width=0.45, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.3,
                 f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_title('P(Disease | Positive Test): Theoretical vs Simulation', fontsize=11, pad=10)
    ax1.set_ylabel('Probability (%)', fontsize=11)
    ax1.set_ylim(0, max(values) * 1.25)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Subplot 2: Breakdown of Positive Test Results in Simulation
    pos_cats = ['True Positives\n(Sick)', 'False Positives\n(Healthy)']
    pos_counts = [sim_results['true_positives'], sim_results['false_positives']]
    colors2 = ['#38b000', '#d90429']

    bars2 = ax2.bar(pos_cats, pos_counts, color=colors2, width=0.45, edgecolor='black')
    for bar in bars2:
        height = bar.get_height()
        pct = (height / sim_results['total_positive_tests']) * 100 if sim_results['total_positive_tests'] > 0 else 0
        ax2.text(bar.get_x() + bar.get_width() / 2, height + (max(pos_counts) * 0.02),
                 f'{height:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_title(f"Breakdown of {sim_results['total_positive_tests']:,} Positive Tests (N={sim_results['num_samples']:,})",
                  fontsize=11, pad=10)
    ax2.set_ylabel('Number of Individuals', fontsize=11)
    ax2.set_ylim(0, max(pos_counts) * 1.25)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle('Bayesian Risk Verification via Monte Carlo Simulation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

