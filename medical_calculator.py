"""
Medical Test Risk Calculator Module using Bayes' Theorem.
Calculates Positive Predictive Value P(Disease | Positive Test)
and Negative Predictive Value P(No Disease | Negative Test).
Includes step-by-step mathematical breakdown and natural language interpretation.
"""


def calculate_bayes_medical(sensitivity: float, specificity: float, prevalence: float) -> dict:
    """
    Calculates P(D|+) and P(~D|-) using Bayes' Theorem.

    Args:
        sensitivity (float): True Positive Rate in percentage (0 to 100).
        specificity (float): True Negative Rate in percentage (0 to 100).
        prevalence (float): Disease Prevalence in percentage (0 to 100).

    Returns:
        dict: Calculated probabilities, step breakdown, and interpretation.
    """
    # Validation
    if not (0.0 <= sensitivity <= 100.0):
        raise ValueError("Sensitivity must be between 0 and 100%.")
    if not (0.0 <= specificity <= 100.0):
        raise ValueError("Specificity must be between 0 and 100%.")
    if not (0.0 <= prevalence <= 100.0):
        raise ValueError("Prevalence must be between 0 and 100%.")

    # Convert percentages to probabilities
    sens = sensitivity / 100.0
    spec = specificity / 100.0
    prev = prevalence / 100.0

    p_d = prev
    p_not_d = 1.0 - p_d

    # P(+ | D) = Sensitivity
    p_pos_given_d = sens
    # P(+ | ~D) = False Positive Rate = 1 - Specificity
    p_pos_given_not_d = 1.0 - spec

    # Denominator: Law of Total Probability for P(+)
    p_pos = (p_pos_given_d * p_d) + (p_pos_given_not_d * p_not_d)

    if p_pos == 0.0:
        raise ZeroDivisionError("Total probability of a positive test P(+) is 0. Cannot compute P(D|+).")

    # Bayes' Theorem: P(D | +) = (P(+ | D) * P(D)) / P(+)
    p_d_given_pos = (p_pos_given_d * p_d) / p_pos

    # P(- | ~D) = Specificity
    p_neg_given_not_d = spec
    # P(- | D) = False Negative Rate = 1 - Sensitivity
    p_neg_given_d = 1.0 - sens

    p_neg = (p_neg_given_not_d * p_not_d) + (p_neg_given_d * p_d)
    p_not_d_given_neg = (p_neg_given_not_d * p_not_d) / p_neg if p_neg > 0 else 0.0

    # Interpretation text
    ppv_pct = p_d_given_pos * 100.0
    interpretation = (
        f"Even though the test has {sensitivity:.1f}% sensitivity and {specificity:.1f}% specificity, "
        f"because the disease is rare (prevalence = {prevalence:.2f}%), a person who tests positive "
        f"actually has only a {ppv_pct:.2f}% chance of truly having the disease. "
        f"The remaining {100.0 - ppv_pct:.2f}% of positive test results are false alarms (false positives)."
    )

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'prevalence': prevalence,
        'p_d': p_d,
        'p_not_d': p_not_d,
        'p_pos_given_d': p_pos_given_d,
        'p_pos_given_not_d': p_pos_given_not_d,
        'p_pos': p_pos,
        'p_d_given_pos': p_d_given_pos,
        'ppv_percentage': ppv_pct,
        'p_not_d_given_neg': p_not_d_given_neg,
        'npv_percentage': p_not_d_given_neg * 100.0,
        'interpretation': interpretation
    }

