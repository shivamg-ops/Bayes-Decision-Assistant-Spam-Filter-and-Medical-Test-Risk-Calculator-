"""
Streamlit Web Application for Bayes Decision Assistant.
Interactive localhost UI covering:
- Module 1: SMS Spam Detection (Scratch Multinomial Naive Bayes)
- Module 2: Medical Test Risk Calculator (Bayes' Theorem & Base Rate Neglect)
- Module 3: Natural Frequency Table (Cohort of 1,000)
- Module 4: Monte Carlo Simulation (100,000 Individuals)
- Module 5: Model Evaluation Metrics & Visualizations
- Module 6: Team Contribution & Project Documentation
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import scratch backend modules
from spam_detection import NaiveBayesClassifier, train_test_split_scratch
from model_evaluation import calculate_metrics, compute_confusion_matrix
from medical_calculator import calculate_bayes_medical
from natural_frequency import generate_natural_frequency_table
from monte_carlo import run_monte_carlo_simulation

# Page Configuration
st.set_page_config(
    page_title="Bayes Decision Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2563EB;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_train_spam_model():
    """Cache data loading and scratch model training for instant interactivity."""
    dataset_path = "data/sms_spam_cleaned.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "../data/sms_spam_cleaned.csv"
    df = pd.read_csv(dataset_path)
    train_df, test_df = train_test_split_scratch(df, test_ratio=0.2, random_state=42)
    model = NaiveBayesClassifier(alpha=1.0)
    model.fit(train_df['message'].tolist(), train_df['label'].tolist())
    y_test = test_df['label'].tolist()
    y_pred = model.batch_predict(test_df['message'].tolist())
    metrics = calculate_metrics(y_test, y_pred, pos_label='spam')
    return df, train_df, test_df, model, metrics


# Load model
try:
    df, train_df, test_df, model, metrics = load_and_train_spam_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/probability.png", width=70)
st.sidebar.title("Bayes Assistant")
st.sidebar.caption("MCA Sem I: Mathematical Foundations Mini Project")

menu_choice = st.sidebar.radio(
    "Select Module:",
    [
        "📩 SMS Spam Detection",
        "🩺 Medical Test Risk Calculator",
        "👥 Natural Frequency Table (1,000)",
        "🎲 Monte Carlo Simulation (100k)",
        "📊 Model Evaluation & Metrics",
        "👥 Team Contribution & Report"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Team Roll Numbers:** 26 to 31  
**Dataset:** UCI SMS Spam Collection  
**Backend:** Pure NumPy/Python (No Sklearn)
""")

# ==========================================================
# MODULE 1: SMS SPAM DETECTION
# ==========================================================
if menu_choice == "📩 SMS Spam Detection":
    st.markdown('<div class="main-header">📩 SMS Spam Detection Filter</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multinomial Naive Bayes implemented from scratch with Laplace Smoothing</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Test an SMS Message")
        sample_messages = [
            "Select a pre-filled sample...",
            "Congratulations! You won a $1,000 Walmart gift card. Click http://claim.com now to win!",
            "Urgent: Your mobile number won 2,000,000 pounds in national lottery! Call 09061743835.",
            "Hey bro, are we meeting at the college library at 4 PM today?",
            "Can you please send me the lecture notes for math when you get home?",
            "Free entry in 2 a weekly competition to win FA Cup final tickets! Text FA to 87121."
        ]
        sample_pick = st.selectbox("Quick-fill with sample messages:", sample_messages)
        default_val = "" if sample_pick == "Select a pre-filled sample..." else sample_pick

        user_input = st.text_area("Enter SMS text to classify:", value=default_val, height=120)

        if st.button("Classify SMS", type="primary"):
            if not user_input.strip():
                st.warning("Please enter a valid text message.")
            else:
                prediction = model.predict(user_input)
                probs = model.predict_proba(user_input)

                st.markdown("---")
                if prediction == "spam":
                    st.error(f"🚨 **PREDICTION: SPAM (Unsolicited / Malicious)**")
                else:
                    st.success(f"✅ **PREDICTION: HAM (Legitimate / Safe)**")

                m_col1, m_col2 = st.columns(2)
                m_col1.metric("P(Spam | Message)", f"{probs['spam']*100:.2f}%")
                m_col2.metric("P(Ham | Message)", f"{probs['ham']*100:.2f}%")

                st.progress(probs['spam'])

    with col2:
        st.subheader("Training Corpus Summary")
        st.write(f"- **Total Training Messages:** {len(train_df):,}")
        st.write(f"- **Test Messages:** {len(test_df):,}")
        st.write(f"- **Vocabulary Size:** {model.vocab_size:,} unique words")
        st.write(f"- **Laplace Smoothing (α):** 1.0 (Add-1 smoothing)")
        st.write(f"- **Test Set Accuracy:** **{metrics['Accuracy']*100:.2f}%**")
        st.write(f"- **F1-Score:** **{metrics['F1-Score']:.4f}**")

# ==========================================================
# MODULE 2: MEDICAL TEST RISK CALCULATOR
# ==========================================================
elif menu_choice == "🩺 Medical Test Risk Calculator":
    st.markdown('<div class="main-header">🩺 Medical Test Risk Calculator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating Base Rate Neglect using Bayes\' Theorem</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Diagnostic Parameters")
        sensitivity = st.slider("Sensitivity / True Positive Rate P(+|D) (%)", 50.0, 100.0, 95.0, 0.5)
        specificity = st.slider("Specificity / True Negative Rate P(-|~D) (%)", 50.0, 100.0, 90.0, 0.5)
        prevalence = st.slider("Disease Prevalence P(D) (%)", 0.05, 20.0, 1.0, 0.05)

        calc = calculate_bayes_medical(sensitivity, specificity, prevalence)

    with col2:
        st.subheader("Bayesian Calculation Results")
        ppv = calc['ppv_percentage']
        npv = calc['npv_percentage']

        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Positive Predictive Value P(D|+)", f"{ppv:.2f}%")
        res_col2.metric("Negative Predictive Value P(~D|-)", f"{npv:.2f}%")

        if ppv < 50.0:
            st.warning(f"⚠️ **The Base Rate Paradox:** Even though the test is **{sensitivity}% sensitive** and **{specificity}% specific**, a positive test only indicates a **{ppv:.2f}%** actual probability of having the disease because the condition is rare ({prevalence}% prevalence). **{100-ppv:.2f}%** of positive results are false alarms!")
        else:
            st.info(f"Because the disease prevalence is higher, the Positive Predictive Value is **{ppv:.2f}%**.")

        with st.expander("Show Step-by-Step Mathematical Proof"):
            st.markdown(f"""
            1. **Prior Probabilities:**
               - $P(D) = {calc['p_d']:.4f}$
               - $P(\\neg D) = {calc['p_not_d']:.4f}$
            2. **Test Likelihoods:**
               - $P(+ \\mid D) = {calc['p_pos_given_d']:.4f}$ (Sensitivity)
               - $P(+ \\mid \\neg D) = {calc['p_pos_given_not_d']:.4f}$ (False Positive Rate = $1 - \\text{{Specificity}}$)
            3. **Law of Total Probability $P(+)$:**
               $$P(+) = P(+ \\mid D)P(D) + P(+ \\mid \\neg D)P(\\neg D) = {calc['p_pos']:.6f}$$
            4. **Bayes' Theorem $P(D \\mid +)$:**
               $$P(D \\mid +) = \\frac{{P(+ \\mid D)P(D)}}{{P(+)}} = \\frac{{{calc['p_pos_given_d']:.4f} \\times {calc['p_d']:.4f}}}{{{calc['p_pos']:.6f}}} = {calc['p_d_given_pos']:.4f} \\implies \\mathbf{{{ppv:.2f}\\%}}$$
            """)

# ==========================================================
# MODULE 3: NATURAL FREQUENCY TABLE
# ==========================================================
elif menu_choice == "👥 Natural Frequency Table (1,000)":
    st.markdown('<div class="main-header">👥 Natural Frequency Table</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explaining Probabilities with Whole Numbers in a Cohort of 1,000 People</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    sens = c1.number_input("Sensitivity (%)", 1.0, 100.0, 95.0, 1.0)
    spec = c2.number_input("Specificity (%)", 1.0, 100.0, 90.0, 1.0)
    prev = c3.number_input("Prevalence (%)", 0.01, 50.0, 1.0, 0.1)

    table = generate_natural_frequency_table(sens, spec, prev, population=1000)

    # Display Table
    freq_df = pd.DataFrame({
        "Category": ["Test Positive (+)", "Test Negative (-)", "Total Cohort"],
        "Disease Present (Sick)": [f"{table['true_positives']:.1f} (True Positives)", f"{table['false_negatives']:.1f} (False Negatives)", f"{table['num_with_disease']:.1f}"],
        "Disease Absent (Healthy)": [f"{table['false_positives']:.1f} (False Positives)", f"{table['true_negatives']:.1f} (True Negatives)", f"{table['num_without_disease']:.1f}"],
        "Total": [f"{table['total_positive_tests']:.1f}", f"{table['total_negative_tests']:.1f}", "1,000"]
    })
    st.table(freq_df)

    st.subheader("Plain English Interpretation:")
    st.markdown(f"""
    - In a typical town of **1,000 people**, only **{table['num_with_disease']:.1f}** individuals actually have the disease.
    - Of those sick people, **{table['true_positives']:.1f}** test positive.
    - Out of the **{table['num_without_disease']:.1f} healthy people**, **{table['false_positives']:.1f}** will falsely test positive!
    - That gives **{table['total_positive_tests']:.1f} total positive tests**.
    - If you test positive, your actual chance of having the disease is:
      $$\\frac{{{table['true_positives']:.1f}}}{{{table['total_positive_tests']:.1f}}} = \\mathbf{{{table['calculated_percentage']:.2f}\\%}}$$
    """)

# ==========================================================
# MODULE 4: MONTE CARLO SIMULATION
# ==========================================================
elif menu_choice == "🎲 Monte Carlo Simulation (100k)":
    st.markdown('<div class="main-header">🎲 Monte Carlo Empirical Simulation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Validating Bayes\' Theorem with 100,000 Simulated Virtual Subjects (NumPy)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Simulation Setup")
        sens_sim = st.number_input("Test Sensitivity (%)", 50.0, 100.0, 95.0, 1.0)
        spec_sim = st.number_input("Test Specificity (%)", 50.0, 100.0, 90.0, 1.0)
        prev_sim = st.number_input("Disease Prevalence (%)", 0.1, 20.0, 1.0, 0.1)
        samples = st.selectbox("Cohort Size", [50000, 100000, 200000], index=1)
        run_btn = st.button("Run Monte Carlo Simulation", type="primary")

    with col2:
        if run_btn:
            with st.spinner("Simulating 100,000 individuals using NumPy random sampling..."):
                sim = run_monte_carlo_simulation(sens_sim, spec_sim, prev_sim, num_samples=samples, random_seed=42)

            st.success("Simulation Complete!")
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Theoretical Bayes PPV", f"{sim['theoretical_ppv_pct']:.2f}%")
            sc2.metric("Monte Carlo Simulated PPV", f"{sim['simulated_ppv_pct']:.2f}%")
            sc3.metric("Absolute Difference", f"{sim['difference_pct']:.4f}%")

            # Plot comparison
            fig, ax = plt.subplots(figsize=(7, 3.5))
            bars = ax.bar(["Theoretical Bayes", f"Monte Carlo (N={samples:,})"],
                          [sim['theoretical_ppv_pct'], sim['simulated_ppv_pct']],
                          color=["#2563EB", "#EF4444"], width=0.4, edgecolor='black')
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.2f}%", ha='center', va='bottom', fontweight='bold')
            ax.set_ylabel("P(Disease | Positive) (%)")
            ax.set_ylim(0, max(sim['theoretical_ppv_pct'], sim['simulated_ppv_pct']) * 1.3)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)
            plt.close()

# ==========================================================
# MODULE 5: MODEL EVALUATION & METRICS
# ==========================================================
elif menu_choice == "📊 Model Evaluation & Metrics":
    st.markdown('<div class="main-header">📊 Model Evaluation & Visualizations</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation Metrics and Confusion Matrix for Scratch Naive Bayes Classifier</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['Accuracy']*100:.2f}%")
    m2.metric("Precision (Spam)", f"{metrics['Precision']*100:.2f}%")
    m3.metric("Recall (Spam)", f"{metrics['Recall']*100:.2f}%")
    m4.metric("F1-Score", f"{metrics['F1-Score']:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix Heatmap")
        matrix = [[metrics['TN'], metrics['FP']], [metrics['FN'], metrics['TP']]]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Pred Ham', 'Pred Spam'],
                    yticklabels=['Actual Ham', 'Actual Spam'], ax=ax)
        ax.set_title("Naive Bayes Confusion Matrix (Test Set N=1,114)")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Dataset Class Distribution")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        counts = df['label'].value_counts()
        bars = ax2.bar(counts.index, counts.values, color=['#3B82F6', '#F97316'], width=0.5, edgecolor='black')
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 50, f"{h:,}", ha='center', va='bottom', fontweight='bold')
        ax2.set_title("SMS Spam Collection (5,574 Messages)")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)
        plt.close()

# ==========================================================
# MODULE 6: TEAM CONTRIBUTION & REPORT
# ==========================================================
elif menu_choice == "👥 Team Contribution & Report":
    st.markdown('<div class="main-header">👥 Team Contribution & Project Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">MCA Semester I Mini Project: Roll Numbers 26 to 31</div>', unsafe_allow_html=True)

    team_data = pd.DataFrame([
        {"Roll No": 26, "Name": "Student 1 (Lead)", "Module Contribution": "System Architecture & Integration (main.py, Streamlit UI)"},
        {"Roll No": 27, "Name": "Student 2", "Module Contribution": "Data Acquisition & Text Preprocessing (Regex, Tokenization)"},
        {"Roll No": 28, "Name": "Student 3", "Module Contribution": "Multinomial Naive Bayes Algorithm from Scratch (Laplace Smoothing)"},
        {"Roll No": 29, "Name": "Student 4", "Module Contribution": "Model Evaluation, Confusion Matrix & Metrics Calculation"},
        {"Roll No": 30, "Name": "Student 5", "Module Contribution": "Bayesian Medical Calculator & 1,000-Person Natural Frequency Table"},
        {"Roll No": 31, "Name": "Student 6", "Module Contribution": "Monte Carlo Simulation (100,000 Samples) & Project Report"}
    ])
    st.table(team_data)

    st.subheader("Academic Abstract")
    st.write("""
    This project unifies statistical probability axioms, conditional probability, Bayes' Theorem, and Monte Carlo empirical simulation.
    It demonstrates how Bayes' Theorem applies to both unstructured text classification (SMS Spam Filter) and clinical decision analysis (Medical Screening).
    Through natural frequency formulation and a 100,000-sample NumPy simulation, the application clearly illustrates **Base Rate Neglect**, showing why accurate tests can result in a low probability of disease when the condition is rare.
    """)

