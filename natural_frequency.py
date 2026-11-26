"""
Natural Frequency Table Module.
Translates conditional probabilities into plain-language natural frequency counts
for a cohort of 1,000 people to overcome base rate neglect.
"""


def generate_natural_frequency_table(sensitivity: float, specificity: float, prevalence: float,
                                     population: int = 1000) -> dict:
    """
    Generates natural frequency counts for a given population size (default: 1,000).

    Args:
        sensitivity (float): Sensitivity percentage (0 to 100).
        specificity (float): Specificity percentage (0 to 100).
        prevalence (float): Prevalence percentage (0 to 100).
        population (int): Cohort size (default: 1,000).

    Returns:
        dict: Table numbers and explanation.
    """
    sens = sensitivity / 100.0
    spec = specificity / 100.0
    prev = prevalence / 100.0

    # Disease status breakdown
    num_with_disease = population * prev
    num_without_disease = population * (1.0 - prev)

    # Test outcome breakdown
    true_positives = num_with_disease * sens
    false_negatives = num_with_disease * (1.0 - sens)

    false_positives = num_without_disease * (1.0 - spec)
    true_negatives = num_without_disease * spec

    total_positive_tests = true_positives + false_positives
    total_negative_tests = false_negatives + true_negatives

    calculated_prob = (true_positives / total_positive_tests) if total_positive_tests > 0 else 0.0

    table_data = {
        'population': population,
        'num_with_disease': num_with_disease,
        'num_without_disease': num_without_disease,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'total_positive_tests': total_positive_tests,
        'total_negative_tests': total_negative_tests,
        'calculated_probability': calculated_prob,
        'calculated_percentage': calculated_prob * 100.0
    }

    return table_data


def format_natural_frequency_display(data: dict) -> str:
    """
    Formats the natural frequency table into an ASCII table and plain-language explanation.
    """
    pop = data['population']
    d_pos = data['num_with_disease']
    d_neg = data['num_without_disease']
    tp = data['true_positives']
    fp = data['false_positives']
    fn = data['false_negatives']
    tn = data['true_negatives']
    tot_pos = data['total_positive_tests']
    prob = data['calculated_percentage']

    output = []
    output.append(f"\nNATURAL FREQUENCY TABLE (Out of {pop:,} People)")
    output.append("+" + "-" * 26 + "+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 15 + "+")
    output.append(f"| {'Category':<24} | {'Disease Present':<18} | {'Disease Absent':<18} | {'Total':<13} |")
    output.append("+" + "-" * 26 + "+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 15 + "+")
    output.append(f"| {'Test Positive (+)':<24} | {tp:14.1f} (TP) | {fp:14.1f} (FP) | {tot_pos:11.1f} |")
    output.append(f"| {'Test Negative (-)':<24} | {fn:14.1f} (FN) | {tn:14.1f} (TN) | {data['total_negative_tests']:11.1f} |")
    output.append("+" + "-" * 26 + "+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 15 + "+")
    output.append(f"| {'Total Cohort':<24} | {d_pos:18.1f} | {d_neg:18.1f} | {pop:11,d} |")
    output.append("+" + "-" * 26 + "+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 15 + "+")

    output.append("\nPlain Words Explanation:")
    output.append(f"- Out of {pop:,} people, {d_pos:.1f} actually have the disease.")
    output.append(f"- Of those {d_pos:.1f} sick individuals, {tp:.1f} will test positive.")
    output.append(f"- Out of {d_neg:.1f} healthy individuals, {fp:.1f} will falsely test positive.")
    output.append(f"- Therefore, a total of {tot_pos:.1f} people receive a positive test result.")
    output.append(f"- Chance of disease given a positive test = {tp:.1f} / {tot_pos:.1f} = {prob:.2f}%.")

    return "\n".join(output)

