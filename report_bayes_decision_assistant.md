# MCA Semester I Mini-Project Report
## Mathematical Foundations with Python

---

### Project Title:
**Bayes Decision Assistant: SMS Spam Filter and Medical Test Risk Calculator**

### Project ID & Allocation:
**Project 5: Bayes Decision Assistant [Roll No. 26 to 31]**

---

### Team Members & Contributions

| Roll Number | Full Name | Academic Role & Specific Technical Contribution |
|:---:|:---|:---|
| **Roll No. 26** | Student 1 (Project Lead) | **System Architecture & Console Integration:** Designed modular structure, input routing, error handlers, and the top-level menu in `main.py`. |
| **Roll No. 27** | Student 2 | **Data Acquisition & NLP Preprocessing:** Extracted UCI SMS Spam Collection dataset, developed text cleaning pipelines (regex, lowercasing, tokenization) in `text_preprocessing.py`, and compiled the data dictionary. |
| **Roll No. 28** | Student 3 | **Naive Bayes Implementation from Scratch:** Programmed Multinomial Naive Bayes classifier in pure Python/NumPy, log-space transformations, and Laplace smoothing in `spam_detection.py`. |
| **Roll No. 29** | Student 4 | **Model Evaluation & Visualization Suite:** Implemented confusion matrix logic, accuracy, precision, recall, F1 metrics from scratch, and created Matplotlib/Seaborn visual routines in `model_evaluation.py`. |
| **Roll No. 30** | Student 5 | **Bayesian Medical Calculator & Natural Frequency:** Coded mathematical expressions for Bayes' Theorem in `medical_calculator.py` and natural frequency cohort breakdown (1,000 people) in `natural_frequency.py`. |
| **Roll No. 31** | Student 6 | **Monte Carlo Simulation & Technical Documentation:** Developed 100,000-sample empirical simulation in `monte_carlo.py`, designed unit testing suite (`tests/test_project.py`), and authored this comprehensive report. |

---

## 1. Introduction

Real-world decisions across computer systems, finance, and healthcare routinely operate under conditions of imperfect information and uncertainty. Quantifying this uncertainty requires formal probability theory. Two pervasive paradigms that embody probabilistic decision-making are:
1. **Automated Content Classification:** Deciding whether an unstructured electronic communication (such as an incoming SMS message) is benign (*ham*) or malicious/unsolicited (*spam*).
2. **Clinical Diagnostic Interpretation:** Evaluating whether an individual testing positive on a screening assay actually possesses the condition or is a false alarm.

The **Bayes Decision Assistant** is an integrated, menu-driven Python application developed as part of the MCA Semester I curriculum (Unit II: Probability Axioms, Conditional Probability, Bayes' Theorem, and NumPy Random Simulations). The project demonstrates that Bayes' Theorem serves as the unifying mathematical foundation for both text classification and medical risk estimation. Crucially, the system illustrates why high-accuracy diagnostic tests can still lead to counter-intuitively low disease probabilities when a condition is rare—a cognitive bias known as **Base Rate Neglect**.

---

## 2. Problem Statement & Objectives

### Problem Statement
An SMS messaging client must autonomously classify incoming text messages under high text ambiguity, while an individual who receives a positive medical screening result must evaluate their true risk of disease. Both problems represent decisions under uncertainty. 

### Core Project Objectives
1. **Scratch Multinomial Naive Bayes:** Implement an SMS spam classification model in pure Python/NumPy without using third-party machine learning libraries (such as `scikit-learn`).
2. **Laplace Add-1 Smoothing:** Formulate Laplace smoothing to eliminate the zero-frequency problem for previously unseen vocabulary terms.
3. **Medical Test Calculator:** Author an exact Bayesian risk calculator that accepts sensitivity, specificity, and disease prevalence, producing Positive Predictive Value (PPV) and Negative Predictive Value (NPV).
4. **Natural Frequency Cohort Representation:** Translate abstract conditional probabilities into intuitive whole-person frequencies for a hypothetical cohort of 1,000 people to eliminate base rate neglect.
5. **Monte Carlo Empirical Validation:** Execute a vectorized simulation of 100,000 synthetic individuals with NumPy random distributions to verify the theoretical convergence of Bayes' formula.
6. **Robust Software Engineering:** Provide automated unit tests, clean modularity, publication-quality plots, and a menu-driven terminal console interface.

---

## 3. Mathematical Foundations

### 3.1 Probability Axioms & Conditional Probability
Let $\Omega$ be the universal sample space. Under Kolmogorov's axioms:
1. $P(E) \ge 0 \quad \forall E \subseteq \Omega$
2. $P(\Omega) = 1$
3. If $E_1, E_2, \dots$ are mutually exclusive, $P(\bigcup_{i=1}^{\infty} E_i) = \sum_{i=1}^{\infty} P(E_i)$

Conditional probability defines the probability of event $A$ given that event $B$ has occurred ($P(B) > 0$):
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

### 3.2 Bayes' Theorem & Total Probability
Since $P(A \cap B) = P(B \mid A) P(A)$, Bayes' Theorem states:
$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

By the Law of Total Probability, if events $A$ and $\neg A$ partition the sample space:
$$P(B) = P(B \mid A)P(A) + P(B \mid \neg A)P(\neg A)$$

### 3.3 Medical Diagnostic Risk & Base Rate Neglect
In clinical testing:
- **Prevalence:** $P(D)$ — Prior probability of having the disease.
- **Sensitivity:** $P(+ \mid D)$ — Probability that the test is positive given that the patient has the disease.
- **Specificity:** $P(- \mid \neg D)$ — Probability that the test is negative given that the patient is healthy.
- **False Positive Rate:** $P(+ \mid \neg D) = 1 - \text{Specificity}$.

The probability that a person actually has the disease given a positive test result is the **Positive Predictive Value (PPV)**:
$$P(D \mid +) = \frac{P(+ \mid D) \cdot P(D)}{P(+ \mid D) \cdot P(D) + P(+ \mid \neg D) \cdot P(\neg D)}$$

**The Paradox:** When prevalence $P(D)$ is very low (e.g., 1%), even with 95% sensitivity and 90% specificity:
$$P(D \mid +) = \frac{0.95 \times 0.01}{(0.95 \times 0.01) + (0.10 \times 0.99)} = \frac{0.0095}{0.0095 + 0.0990} = \frac{0.0095}{0.1085} \approx 8.76\%$$
Thus, over 91% of individuals testing positive are actually healthy!

### 3.4 Multinomial Naive Bayes for SMS Text Classification
Given an SMS message containing word tokens $d = (w_1, w_2, \dots, w_n)$ and target class $c \in \{\text{ham}, \text{spam}\}$:
$$P(c \mid d) = \frac{P(c) P(w_1, \dots, w_n \mid c)}{P(d)}$$

Under the **Naive Bayes conditional independence assumption**:
$$P(w_1, \dots, w_n \mid c) = \prod_{i=1}^n P(w_i \mid c)$$

To avert numerical underflow resulting from cascading products of small floating-point decimals, computations are evaluated in log-space:
$$\log P(c \mid d) \propto \log P(c) + \sum_{i=1}^n \log P(w_i \mid c)$$

**Laplace (Add-1) Smoothing:**
$$P(w_i \mid c) = \frac{\text{count}(w_i, c) + 1}{\sum_{w \in V} \text{count}(w, c) + |V|}$$
where $|V|$ is the total number of distinct words in the training vocabulary.

---

## 4. System Design & Software Architecture

The software is engineered under a modular functional architecture:

```
Bayes-Decision-Assistant/
├── main.py                   --> Console CLI Menu Loop & Dispatcher
├── text_preprocessing.py     --> Text Normalization, Regex Cleaning & Tokenization
├── spam_detection.py         --> Scratch Multinomial Naive Bayes & Train/Test Split
├── model_evaluation.py       --> Confusion Matrix, Metric Math & Plotting
├── medical_calculator.py     --> Bayesian Risk Engine & Step-by-Step Proof
├── natural_frequency.py      --> 1,000-Person Natural Frequency Generator
├── monte_carlo.py            --> 100,000-Individual Vectorized NumPy Simulation
├── data/sms_spam_cleaned.csv --> Cleaned UCI SMS Spam Dataset (5,574 messages)
├── results/                  --> Output Visualizations (3 PNG plots)
├── notebooks/                --> Jupyter Notebook with Mathematical Theory
└── tests/test_project.py     --> Automated Unit Test Suite (10 unit tests)
```

---

## 5. Implementation Details

1. **Preprocessing (`text_preprocessing.py`):** Converts SMS strings to lowercase, strips URLs using regex (`https?://\S+`), removes numeric strings and punctuation, and extracts words of length $\ge 2$.
2. **Naive Bayes (`spam_detection.py`):** Trains word dictionaries per class, calculates class priors, and evaluates posterior likelihoods using stable log-sum additions. Applies softmax over log-scores to return human-interpretable percentage probabilities.
3. **Medical Risk Calculator (`medical_calculator.py`):** Enforces strict domain validation ($0\% \le \text{param} \le 100\%$), guards against zero denominators, and prints explicit intermediate computation steps.
4. **Natural Frequency Table (`natural_frequency.py`):** Computes expected cell counts for a 1,000-person cohort and renders an ASCII grid highlighting True Positives vs False Positives.
5. **Monte Carlo Engine (`monte_carlo.py`):** Generates $10^5$ uniform random floats using `np.random.rand()`. Vectorized boolean arrays assign ground truth infection status and simulate test outcomes without explicit Python loops, achieving execution speeds under 0.05 seconds.

---

## 6. Experimental Results & Visualizations

### 6.1 SMS Spam Classifier Evaluation (Test Set: $N = 1,114$)

The model was trained on 4,460 messages (80%) and evaluated on an independent holdout test set of 1,114 messages (20%):

| Performance Metric | Numerical Value | Academic Interpretation |
|:---|:---:|:---|
| **True Positives (TP)** | 149 | Spam messages correctly caught by the filter |
| **False Positives (FP)** | 20 | Legitimate ham messages misclassified as spam |
| **True Negatives (TN)** | 934 | Benign ham messages correctly recognized |
| **False Negatives (FN)** | 11 | Spam messages that slipped past the filter |
| **Overall Accuracy** | **97.22%** | High overall classification accuracy |
| **Precision (Spam)** | **88.17%** | When flagged as spam, 88.17% are genuinely spam |
| **Recall (Spam)** | **93.13%** | The filter intercepts 93.13% of all spam messages |
| **F1-Score (Spam)** | **0.9058** | Harmonic mean confirming robust balance |

### 6.2 Medical Risk & Monte Carlo Empirical Convergence ($N = 100,000$)

Parameters: Sensitivity = 95.0%, Specificity = 90.0%, Disease Prevalence = 1.0%

| Metric | Theoretical Bayes Formula | Monte Carlo Simulation ($N=100,000$) | Absolute Discrepancy |
|:---|:---:|:---:|:---:|
| **Population Infected** | 1,000 (1.00%) | 1,004 (1.00%) | +4 subjects |
| **True Positives (TP)** | 950 | 959 | +9 subjects |
| **False Positives (FP)** | 9,900 | 9,977 | +77 subjects |
| **Total Positive Tests** | 10,850 | 10,936 | +86 subjects |
| **$P(\text{Disease} \mid +)$ (PPV)** | **8.7558%** | **8.7692%** | **0.0134%** |

**Conclusion:** The Monte Carlo simulation confirms Bayes' Theorem empirically within a margin of error of less than $0.02\%$.

### 6.3 Visualizations Generated
All three figures were programmatically exported to `results/`:
1. `spam_distribution.png`: Bar chart demonstrating the imbalanced distribution of the UCI SMS dataset (86.6% Ham vs 13.4% Spam).
2. `confusion_matrix.png`: Annotated Seaborn heatmap representing the 2x2 contingency matrix on the test dataset.
3. `simulation_graph.png`: Dual-panel figure depicting theoretical vs simulated Positive Predictive Value, along with the physical volume breakdown of true vs false positive test results.

---

## 7. Automated Testing & Verification

The test suite in `tests/test_project.py` consists of 10 automated unit tests utilizing Python's native `unittest` framework:
- `test_medical_probability_calculation`: Verifies classical epidemiological benchmarks.
- `test_invalid_percentage_inputs`: Asserts that out-of-range percentages raise `ValueError`.
- `test_zero_denominator_case`: Asserts that impossible test combinations raise `ZeroDivisionError`.
- `test_natural_frequency_calculations`: Checks arithmetic consistency across the 1,000-person cohort.
- `test_monte_carlo_simulation`: Ensures simulated PPV converges within 1.0% of theoretical Bayes.
- `test_text_preprocessing`: Verifies removal of URLs, punctuation, and phone numbers.
- `test_spam_prediction`: Tests real message classification logic.
- `test_empty_message`: Verifies graceful fallback to prior probabilities without crashing.
- `test_unknown_words`: Validates that out-of-vocabulary words are smoothed via Laplace smoothing.
- `test_metrics_calculation`: Validates scratch confusion matrix formulas against manual calculations.

**Test Run Result:** All 10 tests execute and pass in 0.002 seconds (`Ran 10 tests in 0.002s, OK`).

---

## 8. Limitations & Scope for Future Work

1. **Bag-of-Words Independence Assumption:** Naive Bayes assumes individual words occur independently. It cannot naturally capture grammatical negation phrases (e.g., "no cash required" contains "cash" and may be misjudged).
2. **Vocabulary Shift:** Spammers frequently misspell words intentionally (e.g., `w!n`, `fr33`) to evade filters, requiring n-gram or subword tokenization.
3. **Static Medical Risk Models:** The medical calculator assumes uniform prevalence across all individuals. Clinical practice requires Bayesian updating using hierarchical patient characteristics (age, sex, lifestyle, family history).

---

## 9. Conclusion

The **Bayes Decision Assistant** successfully demonstrates the practical implementation of Bayes' Theorem using pure Python and NumPy. By bridging natural language processing (SMS Spam Detection) and clinical epidemiology (Medical Test Risk Calculator), the project highlights how mathematical foundations enable principled decisions under uncertainty. In particular, the natural frequency table and Monte Carlo simulation demystify Base Rate Neglect, proving both theoretically and empirically that a positive screening result from an accurate test does not equate to a high probability of disease when the condition is rare.

