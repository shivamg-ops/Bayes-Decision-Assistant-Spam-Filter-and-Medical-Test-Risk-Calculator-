# Bayes Decision Assistant: Spam Filter and Medical Test Risk Calculator

**MCA Semester I Mini Project: Mathematical Foundations with Python**  
**Group Roll Numbers:** Roll No. 29

---

## 1. Project Introduction & Objectives

Uncertainty is omnipresent in real-world decision-making. When an SMS filter evaluates whether an incoming text is spam or when a patient receives a positive medical screening result, decisions must be made using probabilistic evidence.

The **Bayes Decision Assistant** is a standalone, menu-driven Python console application designed to apply **Bayes' Theorem** across two distinct domains:

1. **SMS Spam Detection:** An end-to-end Multinomial Naive Bayes text classifier developed entirely from scratch using NumPy (without scikit-learn).
2. **Medical Test Risk Calculator:** A Bayesian risk evaluator that illustrates the counter-intuitive phenomenon of **Base Rate Neglect** (why a highly accurate medical test can yield a low probability of disease when the condition is rare).
3. **Natural Frequency Formulation:** Translates abstract probabilities into intuitive whole-number ratios for a cohort of 1,000 people.
4. **Monte Carlo Simulation:** Simulates 100,000 individuals using NumPy random sampling to empirically validate theoretical Bayesian conclusions.

---

## 2. Technologies Used

- **Language:** Python 3.8+
- **Numerical Computing:** NumPy (arrays, vectorized boolean masks, random sampling)
- **Data Manipulation:** Pandas (CSV loading, train-test splitting)
- **Visualization:** Matplotlib & Seaborn (confusion matrix, distributions, comparison charts)
- **Automated Testing:** Python `unittest` framework
- **No Web Frameworks:** Pure terminal console interface as required by MCA guidelines.

---

## 3. Project Directory Structure

```
Bayes-Decision-Assistant/
│
├── main.py                     # Interactive console menu application
├── spam_detection.py           # Multinomial Naive Bayes classifier from scratch
├── text_preprocessing.py       # Regex text cleaner and tokenizer
├── model_evaluation.py         # Confusion matrix, metrics, and plotting routines
├── medical_calculator.py       # Bayes' theorem medical risk engine
├── natural_frequency.py        # Natural frequency table generator (1,000 people)
├── monte_carlo.py              # Monte Carlo simulation engine (100,000 people)
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation
│
├── data/
│   ├── sms_spam_cleaned.csv    # Cleaned UCI SMS Spam Collection dataset (5,574 rows)
│   └── data_dictionary.md      # Data dictionary and schema specifications
│
├── results/
│   ├── confusion_matrix.png    # Heatmap of test set performance
│   ├── spam_distribution.png   # Dataset class distribution bar chart
│   └── simulation_graph.png    # Theoretical vs Monte Carlo comparison graph
│
├── notebooks/
│   └── mathematical_explanation.ipynb # Jupyter notebook with LaTeX mathematical proofs
│
└── tests/
    └── test_project.py         # 10 comprehensive unit tests covering edge cases
```

---

## 4. Installation and Setup Instructions

### Step 1: Clone or Navigate to the Repository

```bash
cd "Bayes-Decision-Assistant"
```

### Step 2: Create and Activate a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate       # On macOS/Linux
# or venv\Scripts\activate     # On Windows
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. How to Run the Project

### Running the Console Application

```bash
python3 main.py
```

You will see the main menu:

```
========== BAYES DECISION ASSISTANT ==========
1. SMS Spam Detection
2. Medical Test Risk Calculator
3. View Natural Frequency Table
4. Run Monte Carlo Simulation
5. View Model Evaluation Results
6. Generate Visualizations
7. Exit
==============================================
```

### Running the Unit Tests

Execute the comprehensive test suite (10 unit tests):

```bash
python3 -m unittest discover tests
```

### Launching the Mathematical Notebook

```bash
jupyter notebook notebooks/mathematical_explanation.ipynb
```

---

## 6. Mathematical Foundations

### 1. Bayes' Theorem

Given an event $A$ (hypothesis) and event $B$ (evidence):
$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

Using the Law of Total Probability:
$$P(B) = P(B \mid A)P(A) + P(B \mid \neg A)P(\neg A)$$

### 2. Medical Diagnosis Application

- **Sensitivity:** $P(+ \mid D) = \text{True Positive Rate}$
- **Specificity:** $P(- \mid \neg D) = \text{True Negative Rate} \implies P(+ \mid \neg D) = 1 - \text{Specificity}$
- **Prevalence:** $P(D) = \text{Prior probability of disease}$

Positive Predictive Value (PPV):
$$P(D \mid +) = \frac{\text{Sensitivity} \times \text{Prevalence}}{(\text{Sensitivity} \times \text{Prevalence}) + ((1 - \text{Specificity}) \times (1 - \text{Prevalence}))}$$

_Example:_ For a disease with 1% prevalence, 95% sensitivity, and 90% specificity:
$$P(D \mid +) = \frac{0.95 \times 0.01}{(0.95 \times 0.01) + (0.10 \times 0.99)} = \frac{0.0095}{0.0095 + 0.0990} = \frac{0.0095}{0.1085} \approx 8.76\%$$

### 3. Multinomial Naive Bayes for Text

Assuming conditional independence between words given the class $c \in \{\text{ham}, \text{spam}\}$:
$$P(c \mid w_1, w_2, \dots, w_n) \propto P(c) \prod_{i=1}^{n} P(w_i \mid c)$$

To prevent floating-point underflow, computations are performed in log-space:
$$\log P(c \mid w_1, \dots, w_n) = \log P(c) + \sum_{i=1}^{n} \log P(w_i \mid c)$$

**Laplace (Add-1) Smoothing:**
$$P(w \mid c) = \frac{\text{count}(w, c) + \alpha}{\sum_{w' \in V} \text{count}(w', c) + \alpha |V|}$$

---

## 7. Results and Performance Summary

| Metric / Scenario                                              |                 Value                 |
| :------------------------------------------------------------- | :-----------------------------------: | ------ | --------------------- |
| **Total Dataset Size**                                         | 5,574 messages (4,827 Ham / 747 Spam) |
| **Train / Test Split**                                         | 4,460 Train (80%) / 1,114 Test (20%)  |
| \*\*Vocabulary Size ($                                         |                   V                   | $)\*\* | 7,651 distinct tokens |
| **Test Accuracy**                                              |              **97.22%**               |
| **Precision (Spam)**                                           |              **88.17%**               |
| **Recall (Spam)**                                              |              **93.13%**               |
| **F1-Score (Spam)**                                            |              **0.9058**               |
| **Medical Risk Theoretical PPV (Sens=95%, Spec=90%, Prev=1%)** |               **8.76%**               |
| **Monte Carlo Simulated PPV ($N=100,000$)**                    |               **8.77%**               |
| **Empirical Difference**                                       |              **0.0145%**              |

---

## 8. Limitations

1. **Conditional Independence Assumption:** Naive Bayes assumes word occurrences are conditionally independent, ignoring phrase contexts (e.g., "not free").
2. **Out-of-Vocabulary (OOV) Shift:** New slang or deliberately misspelled spam words rely purely on prior class probabilities and smoothing.
3. **Medical Population Homogeneity:** Assumes constant prevalence across all sub-populations without adjusting for age, genetic markers, or lifestyle risk factors.

---

## 9. Team Contributions (Roll No. 26 to 31)

| Roll Number | Team Member Name     | Specific Core Responsibility                                                                                                                                |
| :---------: | :------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   **26**    | Team Member 1 (Lead) | **Project Integration & Console Architecture:** Developed `main.py`, command-line interaction loop, menu dispatching, and system integration.               |
|   **27**    | Team Member 2        | **Dataset Preparation & Preprocessing:** UCI dataset download, cleaning pipeline, regex tokenization in `text_preprocessing.py`, and data dictionary.       |
|   **28**    | Team Member 3        | **Naive Bayes Implementation from Scratch:** Developed `spam_detection.py`, log-likelihood calculations, Laplace smoothing, and probability estimation.     |
|   **29**    | Team Member 4        | **Model Evaluation & Visualizations:** Implemented `model_evaluation.py`, confusion matrix calculations, accuracy, precision, recall, and plotting scripts. |
|   **30**    | Team Member 5        | **Medical Calculator & Natural Frequency:** Developed `medical_calculator.py` and `natural_frequency.py`, mathematical proofs, and step-by-step breakdowns. |
|   **31**    | Team Member 6        | **Monte Carlo Simulation & Documentation:** Implemented `monte_carlo.py` using NumPy vectorization, test suite in `test_project.py`, and academic report.   |
