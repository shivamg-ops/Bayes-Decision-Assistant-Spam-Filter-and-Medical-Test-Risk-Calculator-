import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("""# Mathematical Foundations of Bayes' Theorem
## Project 5: Bayes Decision Assistant (Spam Filter and Medical Test Risk Calculator)
**Course:** MCA Semester I – Mathematical Foundations with Python  
**Team Allocation:** Roll Numbers 26 to 31  

---
### 1. Introduction and Probability Axioms
Probability theory provides a rigorous mathematical framework for quantifying uncertainty.
Let $\\Omega$ be the sample space and $E \\subseteq \\Omega$ be an event. Kolmogorov's probability axioms state:
1. **Non-negativity:** $P(E) \\ge 0$ for every event $E$.
2. **Normalization:** $P(\\Omega) = 1$.
3. **Additivity:** For any countable sequence of mutually exclusive events $E_1, E_2, \\dots$:
$$P\\left(\\bigcup_{i=1}^{\\infty} E_i\\right) = \\sum_{i=1}^{\\infty} P(E_i)$$
"""),

    nbf.v4.new_markdown_cell("""### 2. Conditional Probability and Bayes' Theorem
The conditional probability of an event $A$ occurring given that event $B$ has occurred ($P(B) > 0$) is defined as:
$$P(A \\mid B) = \\frac{P(A \\cap B)}{P(B)}$$

Similarly, $P(B \\mid A) = \\frac{P(A \\cap B)}{P(A)} \\implies P(A \\cap B) = P(B \\mid A)P(A)$. Substituting gives **Bayes' Theorem**:
$$P(A \\mid B) = \\frac{P(B \\mid A) P(A)}{P(B)}$$

Using the **Law of Total Probability**, where $A$ and $\\neg A$ partition the sample space:
$$P(B) = P(B \\mid A)P(A) + P(B \\mid \\neg A)P(\\neg A)$$
Hence:
$$P(A \\mid B) = \\frac{P(B \\mid A)P(A)}{P(B \\mid A)P(A) + P(B \\mid \\neg A)P(\\neg A)}$$
"""),

    nbf.v4.new_markdown_cell("""### 3. Application 1: Medical Test Risk & Base Rate Neglect
Consider a diagnostic test for a condition $D$:
- **Sensitivity:** $P(+ \\mid D)$ (True Positive Rate)
- **Specificity:** $P(- \\mid \\neg D)$ (True Negative Rate) $\\implies P(+ \\mid \\neg D) = 1 - \\text{Specificity}$ (False Positive Rate)
- **Prevalence:** $P(D)$ (Prior Probability of Disease)

The **Positive Predictive Value (PPV)** is:
$$P(D \\mid +) = \\frac{P(+ \\mid D)P(D)}{P(+ \\mid D)P(D) + P(+ \\mid \\neg D)P(\\neg D)}$$
"""),

    nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_bayes(sensitivity, specificity, prevalence):
    sens = sensitivity / 100.0
    spec = specificity / 100.0
    prev = prevalence / 100.0
    
    p_pos_given_d = sens
    p_pos_given_not_d = 1.0 - spec
    
    p_pos = (p_pos_given_d * prev) + (p_pos_given_not_d * (1.0 - prev))
    ppv = (p_pos_given_d * prev) / p_pos
    return ppv * 100.0

# Demonstrate base rate neglect across prevalences (0.1% to 20%)
prevalences = np.linspace(0.1, 20.0, 100)
ppv_curve = [calculate_bayes(95.0, 90.0, p) for p in prevalences]

plt.figure(figsize=(8, 4.5))
plt.plot(prevalences, ppv_curve, color='#2b5c8f', lw=2.5, label='Sensitivity=95%, Specificity=90%')
plt.axvline(1.0, color='r', linestyle='--', label='Rare Disease (Prevalence = 1.0% -> PPV = 8.76%)')
plt.title('Impact of Disease Prevalence on Positive Predictive Value (PPV)')
plt.xlabel('Disease Prevalence (%)')
plt.ylabel('P(Disease | Positive Test) (%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
"""),

    nbf.v4.new_markdown_cell("""### 4. Application 2: Text Classification with Multinomial Naive Bayes
For document classification, each document $d$ consists of word tokens $w_1, w_2, \\dots, w_n$.
The target class $c \\in \\{\\text{ham}, \\text{spam}\\}$. By Bayes' rule:
$$P(c \\mid w_1, \\dots, w_n) = \\frac{P(c) P(w_1, \\dots, w_n \\mid c)}{P(w_1, \\dots, w_n)}$$

The **Naive Bayes Assumption** posits that individual word occurrences are conditionally independent given the class:
$$P(w_1, \\dots, w_n \\mid c) = \\prod_{i=1}^n P(w_i \\mid c)$$

To prevent floating-point underflow, computations are performed in log space:
$$\\log P(c \\mid w_1, \\dots, w_n) \\propto \\log P(c) + \\sum_{i=1}^n \\log P(w_i \\mid c)$$

#### Laplace (Add-1) Smoothing
If a word was never observed in training for class $c$, $P(w_i \\mid c) = 0$, which causes $\\log(0)$ to fail.
We apply Laplace smoothing:
$$P(w_i \\mid c) = \\frac{\\text{count}(w_i, c) + \\alpha}{\\sum_{w \\in V} \\text{count}(w, c) + \\alpha |V|}$$
where $|V|$ is the total number of unique words in the vocabulary and $\\alpha = 1.0$.
"""),

    nbf.v4.new_code_cell("""import sys
sys.path.append('../')
from spam_detection import NaiveBayesClassifier, train_test_split_scratch
from model_evaluation import calculate_metrics

# Load cleaned dataset
df = pd.read_csv('../data/sms_spam_cleaned.csv')
train_df, test_df = train_test_split_scratch(df, test_ratio=0.2, random_state=42)

model = NaiveBayesClassifier(alpha=1.0)
model.fit(train_df['message'].tolist(), train_df['label'].tolist())

y_test = test_df['label'].tolist()
y_pred = model.batch_predict(test_df['message'].tolist())
metrics = calculate_metrics(y_test, y_pred, pos_label='spam')

print(f"Accuracy:  {metrics['Accuracy']*100:.2f}%")
print(f"Precision: {metrics['Precision']*100:.2f}%")
print(f"Recall:    {metrics['Recall']*100:.2f}%")
print(f"F1-Score:  {metrics['F1-Score']:.4f}")
"""),

    nbf.v4.new_markdown_cell("""### 5. Empirical Verification: Monte Carlo Simulation
We simulate 100,000 independent trials using NumPy's pseudo-random number generator.
By the **Strong Law of Large Numbers**, as the sample size $N \\to \\infty$, the empirical sample mean converges almost surely to the theoretical expected value:
$$\\lim_{N \\to \\infty} \\hat{P}_N(D \\mid +) = P(D \\mid +)$$
"""),

    nbf.v4.new_code_cell("""from monte_carlo import run_monte_carlo_simulation

sim = run_monte_carlo_simulation(sensitivity=95.0, specificity=90.0, prevalence=1.0, num_samples=100000, random_seed=42)
print(f"Theoretical Bayes PPV: {sim['theoretical_ppv_pct']:.4f}%")
print(f"Monte Carlo Sim PPV:   {sim['simulated_ppv_pct']:.4f}%")
print(f"Absolute Difference:   {sim['difference_pct']:.4f}%")
""")
]

with open('Bayes-Decision-Assistant/notebooks/mathematical_explanation.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Notebook generated successfully.')

